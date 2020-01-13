import React from 'react';
import Typography from '@material-ui/core/Typography';
import deepPurple from '@material-ui/core/colors/deepPurple';
import orange from '@material-ui/core/colors/orange';
import Highlightable from 'highlightable';
import {getLast, is_valid} from '../utils';

export const PREDICTION = 'prediction';
export const LABEL = 'label';
export const COMBINATION = 'combination';

const BORDER_CONFIG = '4px solid';

const HIGHLIGHT_CONFIGURATION = {
    pred_config: {
        color: deepPurple[400],
        margin: 0,
    },
    label_config: {
        color: orange[200],
        margin: 3,
    },
    combination_config: {

    },
}

function get_config(mode) {
    let config = null;
    if (mode === PREDICTION) {
        const {pred_config} = HIGHLIGHT_CONFIGURATION;
        config = pred_config;
    } else {
        // mode === "label"
        const {label_config} = HIGHLIGHT_CONFIGURATION;
        config = label_config;
    }

    const {color, margin} = config;
    return {color, margin};
}

function construct_range(start, end, word_start, word_end, id=0, range_props={}) {
    if (start >= end) {
        // invalid range
        // start must come strictly before end
        return null;
    }
    const range = {start, end, word_start, word_end, ...range_props};
    range.data = {id};
    return range;
}

export function precomputation_work(example_text) {
    const word_idxes = [];
    const word_starts = [];
    const word_ends = [];
    let word_start = -1;
    let word_index = 0;
    // word_idexes is an array of length example_text
    // that contains at each index the associated word index
    // if example text was split on ' ', and -1 if a space.
    //
    // example:
    //               a cat
    // word_idxes = [0 -1 1 1 1]
    // word_starts = [0, 2];
    // word_ends = [0, 4] 
    for (let i = 0; i < example_text.length; i++) {
        if (example_text[i] == ' ') {
            word_starts.push(word_start + 1);
            word_index++;
            word_start = i;
            if (word_idxes.length > 0 && word_idxes[word_idxes.length - 1] != -1) {
                word_ends.push(word_idxes.length - 1);
            }
            word_idxes.push(-1);
        } else {
            word_idxes.push(word_index);
        }
    }

    if (word_start + 1 < example_text.length) {
        word_starts.push(word_start + 1);
    }

    if (word_idxes.length > 0 && word_idxes[word_idxes.length - 1] != -1) {
        word_ends.push(word_idxes.length - 1);
    }

    return {
        word_idxes,
        word_starts,
        word_ends,
    };
}

export function compute_ranges(example_text, ranges, precomputed_work=null, range_props={}) {
    // given text seperated by a single space
    // and a series of ranges corresponding to words

    // return an array of highlight compatible range objects
    if (precomputed_work == null) {
        precomputed_work = precomputation_work(example_text);
    }

    let word_starts = null;
    let word_ends = null;
    ({word_starts, word_ends, } = precomputed_work);

    const new_ranges = [];
    for (const i in ranges) {
        const range = ranges[i];
        const start_idx = range[0];
        const end_idx = range[1];

        const word_start = word_starts[start_idx];
        const word_end = word_ends[end_idx - 1];
        const t_range = construct_range(word_start, word_end, start_idx, end_idx, -1, range_props);
        if (t_range !== null) {
            new_ranges.push(t_range);
        }
    }
    return new_ranges;
}


export function highlightableText(example_text, ranges, id=0, color='#ffcc80', enabled=false, props) {
    // optimizes this to return a highlightable component only if there is
    // something to highlight
    const new_ranges = ranges.length > 0 ? compute_ranges(example_text, ranges) : [];
    let internal = example_text;
    if (new_ranges.length > 0 ) {
        internal = (
            <Highlightable
                key={'result_table_highlight_' + id + '_' + color}
                ranges={new_ranges}
                enabled={enabled}
                highlightStyle={{
                    backgroundColor: color,
                }}
                text={example_text}
                {...props}
            />
        );
    }
    return (
        <Typography>
            {internal}
        </Typography>
    );
}

export function is_valid_range(range) {
    return (
        is_valid(range) &&
        is_valid(range.start) &&
        is_valid(range.end) &&
        range.start >= 0 &&
        range.end >= 0
    )
}

export function doubleUnderline(letters_node, key) {
    let {color, margin} = get_config(PREDICTION);
    let p_color = color;
    let p_margin = margin;
    ({color, margin} = get_config(LABEL));
    let l_color = color;
    let l_margin = margin;

    return <span
        key={`${key}-span1`}
        style={{
            borderBottom: `${BORDER_CONFIG} ${l_color}`,
            paddingBottom: l_margin,
        }}>
        <span
            key={`${key}-span2`}
            style={{
                borderBottom: `${BORDER_CONFIG} ${p_color}`,
                paddingBottom: p_margin,
            }}>
            {letters_node}
        </span>
    </span>;
}

export function underlineRender(letterNode, range, rangeIndex, span_props={}, key_type='predicted') {
    const {color} = get_config(PREDICTION);
    return <span
        key={`${key_type}_highlight_view_${range.data.id}-${rangeIndex}`}
        style={{
            borderBottom: `${BORDER_CONFIG} ${color}`,
        }}
        {...span_props}>
        {letterNode}
    </span>;
}

export function highlightRender(letterNode, range, rangeIndex, span_props={}, key_type='label') {
    const {color} = get_config(LABEL);
    return <span
        key={`${key_type}_highlight_view_${range.data.id}-${rangeIndex}`}
        style={{
            backgroundColor: color,
        }}
        {...span_props}
        >
        {letterNode}
    </span>;
}

export function combinedRender(letterNode, range, rangeIndex, span_props={}, key_type='combined') {
    const {color} = get_config(LABEL);
    const pred = get_config(PREDICTION);
    return <span
        key={`${key_type}_highlight_view_${range.data.id}-${rangeIndex}`}
        style={{
            borderBottom: `${BORDER_CONFIG} ${pred.color}`,
            // paddingBottom: '1px',
        }}
        {...span_props}
        >
        <span
            key={`${key_type}_inner_highlight_view_${range.data.id}-${rangeIndex}`}
            style={{
                backgroundColor: color,
            }}
        >
            {letterNode}
        </span>
    </span>;
}

export function combinedRender_1(lettersNode, range, rangeIndex, onMouseOverHighlightedWord) {
    // return (
    //     <span
    //         key={`combined_highlight_view_${range.data.id}-${rangeIndex}`}
    //         style={{
    //             // backgroundColor: range.color,
    //             borderBottom: `1px solid ${range.color}`,
    //             paddingBottom: range.margin * 2,
    //         }}
    //         >
    //         {lettersNode}
    //     </span>
    // );
    if (range.margin === 1) {
        // combine case
        return doubleUnderline(lettersNode, `combined_highlight_view_${range.data.id}-${rangeIndex}`);
        // return lettersNode;
    } else {
        // single case
        return (
            <span
                key={`combined_highlight_view_${range.data.id}-${rangeIndex}`}
                style={{
                    // backgroundColor: range.color,
                    borderBottom: `${BORDER_CONFIG} ${range.color}`,
                    paddingBottom: range.margin,
                }}
                >
                {lettersNode}
            </span>
        );
    }
}

function is_overlapping(a_range, b_range) {
    if (a_range === null || b_range === null) {
        return false;
    }

    if (a_range.start > b_range.start) {
        const temp = a_range;
        a_range = b_range;
        b_range = temp;
    }
    // a.start <= b.start


    // case 0:
    // non-overlapping

    // case 1:
    // [0 1 2 3 4 5 ]
    //    a a a a
    //      b b b b
    //    a c c c b

    // case 3:
    // [0 1 2 3 4 5 ]
    //      b b 
    //    a a a a
    //    a c c a

    return (a_range.end > b_range.start);
    // if (a_range.end <= b_range.start) {
    //     // non overlapping
    //     return false;
    // } 
    // // else if (a_range.end > b_range.start) {
    // //     // case 1 and case 2
    // //     return true;
    // // }
    // return true;
}

function _merge_helper(ranges, new_range) {
    // if label range is a conflict with last element
    // merge them and concat merge
    // else add label range
    const last_range = getLast(ranges);
    if (last_range !== null && is_overlapping(last_range, new_range)) {
        const new_ranges = merge_single(last_range, new_range);
        ranges.pop(); // remove last range for merge
        for (let i = 0; i < new_ranges.length; i++) {
            ranges.push(new_ranges[i]);
        }
    } else {
        ranges.push(new_range);
    }

    return ranges;
}

export function merge_ranges(label_ranges, pred_ranges) {
    // 0 1 2 3 4 5 6
    //   p p p 
    //     l l l l
    //   p c c l l
    //           p p
    //   p c c l c p
    let ranges = [];
    const sort_func = function(a, b) {return a.word_start - b.word_start;};
    label_ranges.sort(sort_func);
    pred_ranges.sort(sort_func);

    let li = 0;
    let pi = 0;
    while (li < label_ranges.length && pi < pred_ranges.length) {
        const label_range = label_ranges[li];
        const pred_range = pred_ranges[pi];
        if (label_range.start < pred_range.start) {
            _merge_helper(ranges, label_range);
            li++;
        } else {
            _merge_helper(ranges, pred_range);
            pi++;
        }
    }

    while (li < label_ranges.length) {
        _merge_helper(ranges, label_ranges[li]);
        li++;
    }

    while (pi < pred_ranges.length) {
        _merge_helper(ranges, pred_ranges[pi]);
        pi++;
    }

    return ranges;
}

function _get_id(first_id, second_id) {
    if (typeof first_id === typeof "a" || first_id > 0) {
        return first_id;
    }

    return second_id;
}

function merge_same(a_range, b_range) {
    const ranges = [];
    let first = b_range;
    let second = a_range;
    if (a_range.start < b_range.start) {
        first = a_range;
        second = b_range;
    }
    ranges.push(construct_range(
        first.start,
        second.end, // since its inclusive
        first.word_start,
        second.word_end,
        first.data.id,
        {mode: first.mode},
    ));

    return ranges;
}

function merge_different(a_range, b_range) {
    let first = b_range;
    let second = a_range;
    if (a_range.start < b_range.start) {
        first = a_range;
        second = b_range;
    }

    // 0 1 2
    //   1 2 3
    // or
    // 0 1 2 3
    //   1 2
    // 0 1 2
    // 0 1 2 3
    const ranges = [];
    ranges.push(construct_range(
        first.start,
        second.start - 1, // since its inclusive
        first.word_start,
        second.word_start,
        first.data.id,
        {mode: first.mode},
    ));

    ranges.push(construct_range(
        second.start,
        Math.min(first.end, second.end),
        second.word_start,
        Math.min(first.word_end, second.word_end),
        _get_id(first.data.id, second.data.id),
        {mode: first.mode === second.mode ? first.mode : COMBINATION},
    ));

    if (first.end < second.end) {
        // 0 1 2 first
        //   1 2 3 second
        ranges.push(construct_range(
            first.end,
            second.end,
            first.word_end,
            second.word_end,
            second.data.id,
            {mode: second.mode},
        ));
    } else {
        // 0 1 2 3 first
        //   1 2 second
        ranges.push(construct_range(
            second.end,
            first.end,
            second.word_end,
            first.word_end,
            first.data.id,
            {mode: first.mode},
        ));
    }

    return ranges;
}

export function merge_single(a_range, b_range) {
    if (!is_overlapping(a_range, b_range)) {
        return [];
    }

    if (a_range.mode === b_range.mode) {
        return merge_same(a_range, b_range);
    } else {
        return merge_different(a_range, b_range);
    }
}

function merge(label_range, pred_range) {
    const res = {
        l_range: label_range,
        p_range: pred_range,
        c_range: null,
    }
    if (!is_overlapping(label_range, pred_range)) {
        return res;
    }

    // label_range and pred_range overlap
    let l_range = {};
    let p_range = {};
    let c_range = {};

    // t_range.start = word_start;
    // t_range.end = word_end;
    // t_range.word_start = word_start;
    // t_range.word_end = word_end;
    // t_range.data = {}
    // t_range.data.id = 1;

    // l_range starts first
    if (label_range.start < pred_range.start) {

        // l_range = label_range.start -> pred_range.start
        // c_range = pred_range.start -> label_range.end
        // p_range = label_range.end -> pred_range.end
        l_range = construct_range(
            label_range.start,
            pred_range.start - 1, 
            label_range.word_start,
            pred_range.word_start - 1,
        );
        c_range = construct_range(
            pred_range.start,
            label_range.end, 
            pred_range.word_start,
            label_range.word_end,
        );
        p_range = construct_range(
            label_range.end,
            pred_range.end,
            label_range.word_end,
            pred_range.word_end,
        );
    } else {
        // p_range = p_range.start -> l_range.start
        // c_range = l_range.start -> p_range.end
        // l_range = p_range.end -> l_range.end
        l_range = construct_range(
            pred_range.end, 
            label_range.end,
            pred_range.word_end,
            label_range.word_end,
        );
        c_range = construct_range(
            label_range.start,
            pred_range.end, 
            label_range.word_start,
            pred_range.word_end,
        );
        p_range = construct_range(
            pred_range.start,
            label_range.start - 1,
            pred_range.word_start,
            label_range.word_start - 1,
        );
    }

    return {l_range, p_range, c_range};
}

export function joinRanges(labeled_ranges, pred_ranges, color_spec, precomputed_work) {
    const sort_func = function(a, b) {return b.word_start = a.word_start;};
    labeled_ranges.sort(sort_func);
    pred_ranges.sort(sort_func);

    let label_color, pred_color, combine_color, word_starts, word_ends = null;
    ({label_color, pred_color, combine_color} = color_spec);
    ({word_starts, word_ends, } = precomputed_work);
    const new_ranges = [];
    let ri = 0;
    if (labeled_ranges.length == 0) {
        while (
            ri < pred_ranges.length
        ) {
            const new_p_range = pred_ranges[ri];
            new_p_range.color = pred_color;
            new_p_range.margin = HIGHLIGHT_CONFIGURATION.pred_config.margin;
            new_ranges.push(new_p_range);
            ri++;
        }
    }
    for (const li in labeled_ranges) {
        const curr_l_range = labeled_ranges[li];
        while (
            ri < pred_ranges.length && 
            pred_ranges[ri].start < curr_l_range.end &&
            !is_overlapping(pred_ranges[ri], curr_l_range)
        ) {
            const new_p_range = pred_ranges[ri];
            new_p_range.color = pred_color;
            new_p_range.margin = HIGHLIGHT_CONFIGURATION.pred_config.margin;
            new_ranges.push(new_p_range);
            ri++;
        }
        if (ri >= pred_ranges.length) {
            const new_l_range = curr_l_range;
            new_l_range.color = label_color;
            new_l_range.margin = HIGHLIGHT_CONFIGURATION.label_config.margin;
            new_ranges.push(new_l_range);
            continue;
        }
        if (pred_ranges[ri].start >= curr_l_range.end) {
            // next label_range
            new_ranges.push(curr_l_range);
        } else if (ri < pred_ranges.length) {
            // pred_ranges[ri], curr_l_range overlap
            let l_range, p_range, c_range = null;
            ({l_range, p_range, c_range} = merge(curr_l_range, pred_ranges[ri]));
            if (l_range !== null) {
                l_range.color = label_color;
                l_range.margin = HIGHLIGHT_CONFIGURATION.label_config.margin;
                new_ranges.push(l_range);
            }
            if (p_range !== null) {
                p_range.color = pred_color;
                p_range.margin = HIGHLIGHT_CONFIGURATION.pred_config.margin;
                new_ranges.push(p_range);
            }

            if (c_range !== null) {
                c_range.color = combine_color;
                c_range.margin = 1;
                new_ranges.push(c_range);
            }
        } else {
            // no more potential ri overlap
            const new_l_range = curr_l_range;
            new_l_range.color = label_color;
            new_l_range.margin = HIGHLIGHT_CONFIGURATION.label_config.margin;
            new_ranges.push(new_l_range);
        }
    }

    return new_ranges;
}

export function onTextHighlighted(range, ranges, precomputation_work, highlightCallback) {
    const {
        word_idxes,
        word_starts,
        word_ends,
    } = precomputation_work;

    const start_idx = range.start;
    const end_idx = range.end;
    const start_word = word_idxes[start_idx];
    const end_word = word_idxes[end_idx];
    range.start = word_starts[start_word];
    range.end = word_ends[end_word];
    range.word_start = start_word;
    range.word_end = end_word;
    range.mode = LABEL;
    range.data.id = `range_start_${range.word_start}_end_${range.word_end}`;
    window.getSelection().removeAllRanges();

    if (is_valid_range(range)) {
        const new_ranges = ranges.slice();
        new_ranges.push(range);
        highlightCallback(new_ranges);
    }

    return highlightCallback(ranges);
}

export function combinedHighlightText(
    example_text,
    labeled_ranges,
    pred_ranges,
    enabled=false,
    id=0,
) {
    // optimizes this to return a highlightable component only if there is
    // something to highlight
    const precomputed_work = precomputation_work(example_text);
    const new_labeled_ranges = labeled_ranges.length > 0 ? 
        compute_ranges(example_text, labeled_ranges, precomputed_work, {mode: LABEL}) :
        [];
    const new_pred_ranges = pred_ranges.length > 0 ?
        compute_ranges(example_text, pred_ranges, precomputed_work, {mode: PREDICTION}) :
        [];
    let new_merged_ranges = merge_ranges(new_labeled_ranges, new_pred_ranges);
    new_merged_ranges = new_merged_ranges.filter(x => x);

    let internal = example_text;
    if (new_merged_ranges.length > 0 ) {
        internal = (
            <Highlightable
                key={`result_table_highlight_combined_p${pred_ranges.length}_l${labeled_ranges.length}_${id}`}
                ranges={new_merged_ranges}
                enabled={enabled}
                text={example_text}
                rangeRenderer={smartRenderer}
            />
        );
    }
    return (
        <Typography>
            {internal}
        </Typography>
    );
}

export function smartRenderer(lettersNode, range, rangeIndex) {
    if (range.mode === PREDICTION) {
        return underlineRender(
            lettersNode,
            range,
            rangeIndex,
            null,
            "predicted"
        );
    } else if (range.mode === LABEL) {
        return highlightRender(
            lettersNode,
            range,
            rangeIndex,
            null,
            "label",
        );
    } else if (range.mode === COMBINATION) {
        return combinedRender(
            lettersNode,
            range,
            rangeIndex,
            null,
        );
    }
}