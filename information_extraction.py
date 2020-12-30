from app.extractor.direction_filter_generator import *
from Levenshtein import *
from enum import Enum, auto
from transitions import Machine, State
import numpy as np
import re
from itertools import product, groupby
import time
import copy
from functools import wraps


def debug(fn):
    def wrapper(*args, **kwargs):
        start = time.time()
        try:
            score = fn(*args, **kwargs)
            print("{} cost {}".format(fn.__name__, time.time() - start))
            return score
        except Exception as e:
            print("{} except {}".format(fn.__name__, repr(e)))

    return wrapper


class StringMatcher:
    def _reset_cache(self):
        self._ratio = self._distance = None
        self._opcodes = self._editops = self._matching_blocks = None

    def __init__(self, seq1='', seq2=''):
        self._str1, self._str2 = seq1, seq2
        self._reset_cache()

    def set_seqs(self, seq1, seq2):
        self._str1, self._str2 = seq1, seq2
        self._reset_cache()

    def set_seq1(self, seq1):
        self._str1 = seq1
        self._reset_cache()

    def set_seq2(self, seq2):
        self._str2 = seq2
        self._reset_cache()

    def get_opcodes(self):
        if not self._opcodes:
            if self._editops:
                self._opcodes = opcodes(self._editops, self._str1, self._str2)
            else:
                self._opcodes = opcodes(self._str1, self._str2)
        return self._opcodes

    def get_editops(self):
        if not self._editops:
            if self._opcodes:
                self._editops = editops(self._opcodes, self._str1, self._str2)
            else:
                self._editops = editops(self._str1, self._str2)
        return self._editops

    def get_matching_blocks(self):
        if not self._matching_blocks:
            self._matching_blocks = matching_blocks(self.get_opcodes(),
                                                    self._str1, self._str2)
        return self._matching_blocks

    def ratio(self):
        if not self._ratio:
            self._ratio = ratio(self._str1, self._str2)
        return self._ratio

    def quick_ratio(self):
        if not self._ratio:
            self._ratio = ratio(self._str1, self._str2)
        return self._ratio

    def real_quick_ratio(self):
        len1, len2 = len(self._str1), len(self._str2)
        return 2.0 * min(len1, len2) / (len1 + len2)

    def distance(self):
        if not self._distance:
            self._distance = distance(self._str1, self._str2)
        return self._distance

    def partial_ratio(self, use_length=False):
        blocks = self.get_matching_blocks()
        scores = []
        len1, len2 = len(self._str1), len(self._str2)
        len_ratio = 2 * min(len1, len2) / (len1 + len2) if len1 and len2 else 0
        for block in blocks:
            long_start = block[1] - block[0] if (block[1] - block[0]) > 0 else 0
            long_end = long_start + len(self._str1)
            long_substr = self._str2[long_start:long_end]
            m2 = StringMatcher(self._str1, long_substr)
            r = m2.ratio()
            if use_length:
                scores.append(r * len_ratio)
            else:
                scores.append(r)
        return max(scores)

    def get_partial_ratio_substr(self, use_length=True):
        blocks = self.get_matching_blocks()
        scores = {}
        len1, len2 = len(self._str1), len(self._str2)
        len_ratio = 2 * min(len1, len2) / (len1 + len2) if len1 and len2 else 0
        for block in blocks:
            long_start = block[1] - block[0] if (block[1] - block[0]) > 0 else 0
            long_end = long_start + len(self._str1)
            long_substr = self._str2[long_start:long_end]
            m2 = StringMatcher(self._str1, long_substr)
            r = m2.ratio()
            ratio = len_ratio if use_length else 1
            if r not in scores:
                scores[r * ratio] = long_substr
            elif r in scores and len(scores[r]) < len(long_substr):
                scores[r * ratio] = long_substr
        key = max(scores.keys())
        return key, scores[key]


class TextBox(Box):
    def __init__(self, text, x1, y1, x2, y2, original_np):
        super(TextBox, self).__init__(x1, y1, x2, y2)
        self.text = text
        self.split_flag = False
        self.original_np = original_np
        self.text_original = "".join(self.decode_score(original=True))
        self.decode_text = "".join([siamese_decode[char] if char in siamese_decode else char for char in self.text])

    def _get_distance(self, textbox):
        return (self.x - textbox.x) ** 2 + (self.y - textbox.y) ** 2

    def get_nearest(self, text_box_list):
        distance = {i: self._get_distance(textbox) for i, textbox in enumerate(text_box_list)}
        return text_box_list[min(distance, key=distance.get)]

    def _get_loc_by_index(self, length, index_start, index_end):
        return int(self.x1 + index_start * self.width / length), int(self.x1 + index_end * self.width / length)

    def _get_np_index(self, index, index_end=None):
        start = 0 if index == 0 else len(
            re.findall("卍{0,}[" + "]{1,}卍{0,}[".join(list(self.text[:index])) + "]{1,}",
                       self.text_original[ctc_padding:])[0]) + ctc_padding
        if index_end:
            end = len(self.original_np) if index == len(self.text) - 1 else len(
                re.findall("卍{0,}[" + "]{1,}卍{0,}[".join(list(self.text[:index_end])) + "]{1,}",
                           self.text_original[ctc_padding:])[0]) + ctc_padding
        else:
            end = len(self.original_np) if index == len(self.text) - 1 else len(
                re.findall("卍{0,}[" + "]{1,}卍{0,}[".join(list(self.text[:index + 1])) + "]{1,}",
                           self.text_original[ctc_padding:])[0]) + ctc_padding
        return start, end

    def decode_score(self, mask=np.array([]), original=False):
        def labels_to_text(labels, join=""):
            ret = []
            for c in labels:
                if c == len(alphabet):  # CTC Blank
                    ret.append(join)
                else:
                    ret.append(alphabet[c])
            return "".join(ret)

        def decode(out, mask):
            ret = []
            for j in range(out.shape[0]):
                out_best = list(np.argmax(out[j, ctc_padding:] * mask, 1)) if mask.any() else list(
                    np.argmax(out[j, ctc_padding:], 1))
                out_best = [k for k, g in groupby(out_best)]
                out_str = labels_to_text(out_best)
                ret.append(out_str)
            return ret[0]

        def decode_original(out):
            ret = []
            out_best = list(np.argmax(out, 1))
            out_str = labels_to_text(out_best, "卍")
            ret.append(out_str)
            return ret

        if original:
            return decode_original(self.original_np)
        elif not self.text:
            return self.text
        elif isinstance(mask, list):
            text = ""
            terminal = 0
            for start, end, mask_np in mask:
                start_, end_ = self._get_np_index(start, end)
                # print("bingo",start, end,start_, end_)
                text += decode(np.expand_dims(self.original_np[start_:end_], 0), mask_np)
                terminal = end_
                # print("bingo",text,len(self.original_np))
            text += decode(np.expand_dims(self.original_np[terminal:], 0), mask_np)
            return text
        elif mask.any():
            return decode(np.expand_dims(self.original_np, 0), mask)
        else:
            return self.text

    def update(self, text, x1, y1, x2, y2, original_np):
        self.text = text
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.box = (x1, y1, x2, y2)
        self.width = abs(x1 - x2)
        self.height = abs(y1 - y2)
        self.center, (self.x, self.y) = self._get_center(), self._get_center()
        self.original_np = original_np
        self.text_original = "".join(self.decode_score(original=True))
        self.decode_text = "".join([siamese_decode[char] if char in siamese_decode else char for char in self.text])

    def split(self, bool_undo_decode, substr):
        # print(self.split_flag,self.text,self.decode_text, bool_undo_decode, substr)
        start = self.text.find(substr) if bool_undo_decode else self.decode_text.find(substr)
        end = start + len(substr)
        np_start, np_end = self._get_np_index(start, end)
        loc_start, loc_end = self._get_loc_by_index(len(self.text), start, end)
        before_box = TextBox(self.text[:start], self.x1, self.y1, loc_start, self.y2, self.original_np[:np_start, :])
        end_box = TextBox(self.text[end:], loc_end, self.y1, self.x2, self.y2,
                          np.concatenate([self.original_np[:ctc_padding, :], self.original_np[np_end:, :]], axis=0))
        mid_box = {'text': self.text[start:end], 'x1': loc_start, 'y1': self.y1, 'x2': loc_end, 'y2': self.y2,
                   'original_np': np.concatenate(
                       [self.original_np[:ctc_padding, :], self.original_np[np_start:np_end, :]], axis=0)}
        self.split_flag = True
        return before_box, mid_box, end_box

    def get_siamese_rate(self, codes, boolean_anchor, siamese_threshold=0.6, legal_length=6):
        if boolean_anchor:
            rate = [StringMatcher(code, self.text).partial_ratio() for code in codes]
            return max(rate) if rate and max(rate) >= siamese_threshold else 0
        else:
            rate = [StringMatcher(code, self.decode_text).partial_ratio(True) for code in codes if
                    abs(len(code) - len(self.decode_text)) <= legal_length]
            return max(rate) if rate else 0

    def get_siamese_substr(self, codes, boolean_anchor):
        strings = [(code, StringMatcher(code, self.text).get_partial_ratio_substr()) for code in
                   codes] if boolean_anchor else [
            (code, StringMatcher(code, self.decode_text).get_partial_ratio_substr()) for code in codes]
        score = {}
        for code, (rate, sub_str) in strings:
            # print("bingo", code, rate, sub_str)
            if rate not in score:
                score[rate] = (sub_str, code)
            elif rate in score and len(score[rate][0]) < len(sub_str):
                score[rate] = (sub_str, code)
        # print("$",max(score.keys()),score[max(score.keys())])
        return score[max(score.keys())]


class Field(object):
    def __init__(self, decode, field):
        self.field = field
        self.decode = decode
        self.use_loc_mask = True if field.startswith("$") else False
        self.boolean_anchor = True if field.startswith("__") else False
        self.siamese_decode = self._get_decode(self.decode)
        self.siamese_code = None
        self.mask = self._get_mask_array()

    def _get_decode(self, decode):
        siamese_decode = []
        for code in decode:
            if "}" in code:
                sub_code = [i for i in code.split("}") if i]
                code_list = []
                for x in sub_code:
                    if "{" in x:
                        before, after = [i for i in x.split("{") if i]
                        if "," in after:
                            min_, max_ = after.split(",")
                            code_list.append([before[:-1] + before[-1] * i for i in range(int(min_), int(max_) + 1)])
                        else:
                            code_list.append([before[:-1] + before[-1] * int(after)])
                    else:
                        code_list.append([x])
                for pairs in product(*code_list):
                    siamese_decode.append("".join(pairs))
            else:
                siamese_decode.append(code)
        return siamese_decode

    def _get_mask_by_chars(self, chars):
        legal = "".join(
            [siamese_decode_config_revert[char] if char in siamese_decode_config_revert else char for char in chars])
        score = np.zeros((len(alphabet) + 1,), dtype=int)
        for i, x in enumerate(alphabet):
            if x in legal:
                score[i] = 1
            else:
                score[i] = 0.01
        score[-1] = 1
        return score

    def _get_mask_array(self):
        chars = set("".join(self.siamese_decode))
        return self._get_mask_by_chars(chars)

    def recount_mask_array(self):
        if self.use_loc_mask and self.siamese_code:
            # print(self.siamese_code)
            self.mask = [(i, i + 1, self._get_mask_by_chars(char)) for i, char in enumerate(self.siamese_code)]

    def siamese_substr(self, textbox):
        substr, self.siamese_code = textbox.get_siamese_substr(self.siamese_decode, self.boolean_anchor)
        # print(self.field, substr, self.siamese_code)
        # print(type(self.mask))
        return self.boolean_anchor, substr

    def siamese_ratio(self, textbox):
        ratio = textbox.get_siamese_rate(self.siamese_decode, self.boolean_anchor)
        # print(self.field, substr, self.siamese_code)
        # print(type(self.mask))
        return ratio

    def prepare(self, text_boxes):
        self.text_boxes = text_boxes
        self.vote_siamese = np.zeros(len(text_boxes), dtype=np.float)
        self.vote_direction = np.zeros(len(text_boxes), dtype=np.float)

    def get_vote_box(self, siamese_weight=0.618, direction_weight=0.372):
        if np.max(self.vote_siamese) == 0:
            return None
        self.vote = self.vote_siamese * siamese_weight + self.vote_direction * direction_weight
        max_ = np.max(self.vote)
        return self.text_boxes[np.where(self.vote == max_)[0][0]] if max_ > 0 else None

    # @debug
    def siamese_progress(self):
        for i, box in enumerate(self.text_boxes):
            self.vote_siamese[i] = box.get_siamese_rate(self.siamese_decode, self.boolean_anchor)

    def direction_progress(self, anchor_weight, direction_filter, boxes_direction):
        #print(direction_filter)
        for j, box in enumerate(self.text_boxes):
            ratio = list(
                map(lambda x, y: 1 + anchor_weight * self.boolean_anchor if x == y else 0, boxes_direction[box],
                    direction_filter))
            # print("ratio:",ratio)
            self.vote_direction[j] = sum(ratio) / len(ratio) if ratio else 0


    def get_siamese_boxes(self, siamese_threshold):
        return [box for i, box in enumerate(self.text_boxes) if self.vote_siamese[i] > siamese_threshold]


class States(Enum):
    prepare = auto()
    init = auto()
    anchor_failure = auto()
    anchor_done = auto()
    field_progress = auto()
    split_box = auto()
    split_box_done = auto()
    field_done = auto()
    terminal = auto()


class DataHandle(object):
    transitions = [{'trigger': 'prepare', 'source': States.prepare, 'dest': States.init, 'prepare': '_prepare'},

                   {'trigger': 'anchor_progress', 'source': States.init, 'dest': States.anchor_failure,
                    'conditions': '_reverse_check_anchor', 'prepare': '_anchor_progress'},
                   {'trigger': 'anchor_progress', 'source': States.init, 'dest': States.anchor_done,
                    'conditions': '_check_anchor', 'prepare': '_anchor_progress'},

                   {'trigger': 'field_progress', 'source': [States.anchor_done, States.field_progress],
                    'dest': States.field_progress, 'conditions': '_reverse_check_field', 'prepare': '_field_progress'},
                   {'trigger': 'field_progress', 'source': [States.anchor_done, States.field_progress],
                    'dest': States.split_box, 'conditions': '_check_field', 'prepare': '_field_progress'},

                   {'trigger': 'field_progress', 'source': States.split_box, 'dest': States.split_box_done,
                    'prepare': '_split_box_prepare_again'},

                   {'trigger': 'field_progress', 'source': States.split_box_done, 'dest': States.split_box_done,
                    'conditions': '_reverse_check_field', 'prepare': '_field_progress'},
                   {'trigger': 'field_progress', 'source': States.split_box_done, 'dest': States.field_done,
                    'conditions': '_check_field', 'prepare': '_field_progress'},

                   {'trigger': 'terminal', 'source': States.field_done, 'dest': States.terminal,
                    'prepare': '_terminal'}]

    def __init__(self, ocr, box, ocr_original, invoice_type, invoice_direction_filter, debug=False, debug_filter=None,
                 bool_require=True, double_fix=False, x_threshold=2, y_threshold=2):
        self.ocr = ocr.copy()
        self.box = box.copy()
        self.ocr_original = ocr_original.copy()
        self.invoice_type = invoice_type if invoice_type in invoice_pattern else "common"
        self.debug = debug
        self.debug_filter = debug_filter
        self.bool_require = bool_require
        self.double_fix = double_fix
        self.x_threshold = x_threshold
        self.y_threshold = y_threshold
        self.text_boxes = []
        self.data = {}
        self.vote_siamese = pd.DataFrame()
        self.vote_direction = pd.DataFrame()
        self.current_score = {}
        self.tries = 0
        self.requirement = invoice_pattern[self.invoice_type]
        self.special_handle = special_handle.get(self.invoice_type, {})
        self.direction_filter = invoice_direction_filter.get(self.invoice_type, {})
        self.output_handle = output_handle.get(self.invoice_type, {})
        self.machine = Machine(model=self, states=States, transitions=DataHandle.transitions, initial=States.prepare)

    def _prepare(self):
        self.text_boxes = [TextBox(text, *self.box[i], self.ocr_original[i]) for i, text in enumerate(self.ocr)]
        self.data = {field: Field(decode, field) for field, decode in self.requirement.items()}
        for field_obj in self.data.values():
            field_obj.prepare(self.text_boxes)
        self.boxes_direction = self._generate_boxes_direction()

    def _generate_boxes_direction(self):
        return {(box, box_): box.get_direction(box_) for box in self.text_boxes for box_ in self.text_boxes}

    # @debug
    def _vote_siamese(self):
        for field_obj in self.data.values():
            field_obj.siamese_progress()

    # @debug
    def _vote_direction(self, current_score, anchor_weight=1):
        if self.direction_filter:
            boxes_direction = {}
            for j, box in enumerate(self.text_boxes):
                boxes_direction_ = []
                for field, box_ in current_score.items():
                    boxes_direction_.append(self.boxes_direction[(box_, box)])
                boxes_direction[box] = boxes_direction_

            direction_filter = {}
            for field_ in self.data:
                direction_filter_ = []
                for field, box_ in current_score.items():
                    direction_filter_.append(self.direction_filter[(field, field_)])
                direction_filter[field_] = direction_filter_

            for field, field_obj in self.data.items():
                field_obj.direction_progress(anchor_weight, direction_filter[field], boxes_direction)

    # @debug
    def _get_current_score(self, limit_anchor=False):
        score = {}
        for field, field_obj in self.data.items():
            text_box = field_obj.get_vote_box()
            # print(field, np.max(field_obj.vote_siamese))
            if field.startswith("__") and limit_anchor and text_box:
                score[field] = text_box
            elif not limit_anchor and text_box:
                score[field] = text_box
        # print({k: v.text for k, v in score.items()})
        return score

    def _anchor_progress(self):
        self._vote_siamese()
        self._vote_direction(self._get_current_score(True))

    def _field_progress(self):
        now = time.time()
        self._vote_direction(self._get_current_score())
        # print('once direction_filter:', time.time() - now)

    def _split_box_prepare_again(self):
        # print({k: v.text for k, v in self.current_score.items()})
        # self.summary("__买票到12306")
        for field, textbox in self.current_score.items():
            # print("###", field, textbox.text, textbox.split_flag)
            bool_undo_decode, substr = self.data[field].siamese_substr(textbox)
            if textbox and textbox.text and not textbox.split_flag:
                # print(substr)
                before, mid, after = textbox.split(bool_undo_decode, substr)
                textbox.update(**mid)
                # self.text_boxes.append(TextBox(**mid))
                self.text_boxes.append(before)
                self.text_boxes.append(after)
        for field_obj in self.data.values():
            field_obj.prepare(self.text_boxes)
            field_obj.recount_mask_array()
        self.boxes_direction = self._generate_boxes_direction()
        self._vote_siamese()
        self._vote_direction(self._get_current_score())

    def _concat_text_boxes(self, text_list):
        text, x, y, np_array = [], [], [], []
        for index, i in enumerate(text_list):
            text.append(i.text)
            x.append(i.x1)
            x.append(i.x2)
            y.append(i.y1)
            y.append(i.y2)
            np_array.append(i.original_np[ctc_padding:]) if index > 0 else np_array.append(i.original_np)
        return "".join(text), min(x), min(y), max(x), max(y), np.concatenate(np_array, axis=0)

    def _handle_by(self, handle, text_box_list, current_score, field, anchor, siamese_threshold=0.55):
        x = {i: textbox.x for i, textbox in enumerate(text_box_list)}
        y = {i: textbox.y for i, textbox in enumerate(text_box_list)}
        if handle == "concat_x":
            dict_sorted = sorted(x.items(), key=lambda i: i[1], reverse=False) if len(x) > 1 else list(x.items())
            text_list = [text_box_list[i] for i, loc in dict_sorted]
            text_list = [box for box in text_list if self.data[field].siamese_ratio(box) > siamese_threshold]
            return TextBox(*self._concat_text_boxes(text_list))
        elif handle == "concat_y":
            dict_sorted = sorted(y.items(), key=lambda i: i[1], reverse=False) if len(y) > 1 else list(y.items())
            text_list = [text_box_list[i] for i, loc in dict_sorted]
            text_list = [box for box in text_list if self.data[field].siamese_ratio(box) > siamese_threshold]
            return TextBox(*self._concat_text_boxes(text_list))
        elif handle == "drop_illegal":
            return current_score if current_score in text_box_list else None
        elif handle == "nearest":
            boxes = self.data[field].get_siamese_boxes(siamese_threshold)
            if boxes:
                return anchor.get_nearest(boxes)
            return current_score

    def _get_fist_anchor(self, anchors):
        if not isinstance(anchors, list):
            anchors = [anchors]
        for anchor in anchors:
            if anchor in self.current_score:
                return self.current_score[anchor]
        return None

    def _special_handle(self):
        for field, (anchors, direction, handle) in self.special_handle.items():
            anchor_box = self._get_fist_anchor(anchors)
            if anchor_box:
                if isinstance(direction, tuple):
                    box_list = [box for box in self.text_boxes if self.boxes_direction[(anchor_box, box)] in direction]
                else:
                    box_list = [box for box in self.text_boxes if self.boxes_direction[(anchor_box, box)] == direction]
                # print(field, anchors, direction, handle)
                # print(self.current_score[field].text)
                self.current_score[field] = self._handle_by(handle, box_list, self.current_score.get(field, None),
                                                            field, anchor_box) if box_list else None
                # print(self.current_score[field].text)

    def _text_handle(self, text, handle):
        if handle:
            command, loc, char = handle
            if char not in text and command == "insert":
                return text[:loc] + char + text[loc:]
            if char in text and command == "slice":
                index = text.index(char)
                if loc == "before":
                    return text[:index]
                elif loc == "after":
                    return text[index + 1:]
        return text

    def _output_handle(self, text, handles):
        text_ = copy.copy(text)
        for handle in handles:
            # print(text_)
            text_ = self._text_handle(text_, handle)
            # print("@",text_)
        return text_

    def summary(self, key=None):
        if key and key in self.current_score:
            print(self.current_score[key].text)
        print({k: v.text for k, v in self.current_score.items()})

    def _terminal(self):
        # self.summary("terminal")
        self._special_handle()
        # print(self.direction_filter[("$train_number","start")])
        # print(self.direction_filter[("start","$train_number")])
        # self.summary("terminal")
        terminal = {}
        for field, field_obj in self.data.items():
            if field in self.current_score and self.current_score[field]:
                # print(type(field_obj.mask))
                terminal[field] = self.current_score[field].decode_score(field_obj.mask)
                # print(terminal[field])
                terminal[field] = self._output_handle(terminal[field], self.output_handle.get(field, []))
            else:
                terminal[field] = ""
        return terminal

    def _check_anchor(self, least=1):
        score = self._get_current_score(True)
        if len(score) >= least:
            self.current_score = score
            return True
        else:
            return False

    def _reverse_check_anchor(self):
        return not self._check_anchor()

    def _check_field(self):
        score = self._get_current_score()
        self.tries += 1
        if self.current_score == score:
            return True
        elif self.tries > max_try:  # 10次必进box分割
            self.tries = 0
            return True
        else:
            self.current_score = score
            return False

    def _reverse_check_field(self):
        return not self._check_field()

    def extract(self):
        now = time.time()
        self.prepare()
        # print("phase1:", time.time() - now)
        self.anchor_progress()
        # print("phase2:", time.time() - now)
        if self.state == States.anchor_failure:
            return "Failed", None
        else:
            # print("phase3:", time.time() - now)
            while True:
                self.field_progress()
                if self.state == States.field_done:
                    break
            # print("phase4:", time.time() - now)
            terminal = self._terminal()
            return "Succeeded", terminal
