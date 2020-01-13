import React from 'react';
import PropTypes from 'prop-types';
import shallowCompare from 'react-addons-shallow-compare'; 
import Highlightable from 'highlightable';
import CustomHighlightable from './highlightable_text';
import {LABEL, PREDICTION, compute_ranges, merge_ranges,highlightableText, combinedHighlightText, precomputation_work, is_valid_range, smartRenderer} from '../../utils/highlight_utils';
import {is_valid} from '../../utils';


export function remove(range, ranges) {
    if(!ranges || !ranges.length) {
      return ranges;
    }
    
    const immutableRange = ranges.find(r => r.data.id === range.data.id);
    const index = ranges.indexOf(immutableRange);
  
    if (index !== -1) {
        return ranges.splice(index, 1);
    }
    return ranges;
}

class TableHighlightCell extends React.Component {
    state = {
        precomputed_work: null,
        label_ranges: [],
    };



    reset_range(range) {
        const {label_ranges} = this.state;
        let new_ranges = label_ranges.slice();
        remove(range, new_ranges);
        this.setState({
            label_ranges: new_ranges,
        });
    }

    componentWillReceiveProps(newProps) {
        if (is_valid(newProps)) {
            const {label_ranges, example_text} = newProps;
            const precomputed_work = precomputation_work(example_text);
            this.setState({label_ranges, precomputed_work});
        }
    }

    onTextHighlighted(range) {
        const {label_ranges, precomputed_work} = this.state;
        if (!is_valid(precomputed_work)) {
            return;
        }
        const {
            word_idxes,
            word_starts,
            word_ends,
        } = precomputed_work;
    
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
            const new_ranges = label_ranges.slice();
            new_ranges.push([range.word_start, range.word_end + 1]);
            this.setState({label_ranges: new_ranges});
        }
    }

    render() {
        const {id, pred_ranges, example_text, enabled} = this.props;
        const {label_ranges, precomputed_work} = this.state;

        const new_labeled_ranges = label_ranges.length > 0 ? 
            compute_ranges(example_text, label_ranges, precomputed_work, {mode: LABEL}) :
            [];

        const new_pred_ranges = pred_ranges.length > 0 ?
            compute_ranges(example_text, pred_ranges, precomputed_work, {mode: PREDICTION}) :
            [];

        let new_merged_ranges = merge_ranges(new_labeled_ranges, new_pred_ranges);
        new_merged_ranges = new_merged_ranges.filter(x => x);

        return <CustomHighlightable
            key={'result_table_highlight_combined_' + id}
            ranges={new_merged_ranges}
            enabled={enabled}
            text={example_text}
            rangeRenderer={smartRenderer}
            onTextHighlighted={this.onTextHighlighted.bind(this)}
        />;
    }

}

TableHighlightCell.propTypes = {
    id: PropTypes.number,
    label_ranges: PropTypes.array,
    pred_ranges: PropTypes.array,
    example_text: PropTypes.string,
    enabled: PropTypes.bool,
};


export default TableHighlightCell;
