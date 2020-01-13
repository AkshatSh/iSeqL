import React from 'react';
import PropTypes from 'prop-types';
import shallowCompare from 'react-addons-shallow-compare'; 
import Highlightable from 'highlightable';
import {is_valid} from '../../utils';

function compare_range(rangeA, rangeB) {
    if (!is_valid(rangeA) && !is_valid(rangeB)) {
        return true;
    }

    if (rangeA.mode !== rangeB.mode) {
        return false;
    }

    if (rangeA.data.id !== rangeB.data.id) {
        return false;
    }

    if (rangeA.start !== rangeB.start) {
        return false;
    }

    if (rangeA.end !== rangeB.end) {
        return false;
    }

    return true;
}

function compare_ranges(rangeA, rangeB) {
    if (!(is_valid(rangeA) && is_valid(rangeB))) {
        return true;
    }

    if (rangeA.length !== rangeB.length) {
        return true;
    }

    let all_same = true;
    for (let i = 0; i < rangeA.length && all_same; i++) {
        const a_range = rangeA[i];
        const b_range = rangeB[i];
        if (!compare_range(a_range, b_range)) {
            all_same = false;
        }
    }

    return all_same;

}

class CustomHighlightable extends Highlightable {
    shouldComponentUpdate(nextProps, nextState) {
        const testA = shallowCompare(this, nextProps, nextState);
        const testB = compare_ranges(this.props.ranges, nextProps.ranges);
        return testA || testB;
    }

}


export default CustomHighlightable;
