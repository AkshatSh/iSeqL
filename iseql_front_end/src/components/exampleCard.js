import React from 'react';
import PropTypes from 'prop-types';
import Button from '@material-ui/core/Button';
import Typography from '@material-ui/core/Typography';
import { withStyles } from '@material-ui/core/styles';
import Card from '@material-ui/core/Card';
import CardActions from '@material-ui/core/CardActions';
import CardContent from '@material-ui/core/CardContent';
import Highlightable from 'highlightable';
import withRoot from '../withRoot';
import IconButton from '@material-ui/core/IconButton';
import grey from '@material-ui/core/colors/grey';
import CheckCircle from '@material-ui/icons/CheckCircle';
import Clear from '@material-ui/icons/Clear';
import Divider from '@material-ui/core/Divider';
import Tooltip from '@material-ui/core/Tooltip';
import Menu from '@material-ui/core/Menu';
import MenuItem from '@material-ui/core/MenuItem';
import MoreHorizIcon from '@material-ui/icons/MoreHoriz';
import CustomHighlightable from './labeling/highlightable_text';
import {is_valid_range, compute_ranges, underlineRender, highlightRender, combinedRender, merge_ranges, PREDICTION, LABEL, COMBINATION} from '../utils/highlight_utils';
import {getLast, set_array_props, is_valid} from '../utils';
// import Tooltip from 'rc-tooltip';

export const HIGHLIGHT_RANGE = 'HIGHLIGHT_RANGE';
export const REMOVE_HIGHLIGHTED_RANGE = 'REMOVE_HIGHLIGHTED_RANGE';
export const RESET_HIGHLIGHTED_RANGE = 'RESET_HIGHLIGHTED_RANGE';

const ITEM_HEIGHT = 48;

export function highlightRange(range){
    return {type: HIGHLIGHT_RANGE, range};
}

export function removeHighlightRange(range) {
    return {type: REMOVE_HIGHLIGHTED_RANGE, range};
}

export function resetHighlightRange() {
    return {type: RESET_HIGHLIGHTED_RANGE};
}

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

const styles = theme => ({
    card: {
        minWidth: 275,
        marginLeft: 0,
        marginRight: 10,
        marginTop: 10,
        marginBottom: 10,
    },
    noPadding: {
        padding: 0,
    },
    root: {
        textAlign: 'center',
        paddingTop: theme.spacing.unit * 20,
    },
    nested: {
        paddingLeft: theme.spacing.unit * 4,
    },
    fab: {
        margin: theme.spacing.unit,
    },
    bullet: {
        display: 'inline-block',
        margin: '0 2px',
        transform: 'scale(0.8)',
    },
    highlight: {
        backgroundColor: '#ffcc80'
    },
    button: {
        margin: theme.spacing.unit,
    },
});

class ExampleCard extends React.Component {
    state = {
        label: null,
        ranges: [],
        pred_ranges: [],
        word_indexes: [],
        show_ranges: [],
        iteration: 0,

        // menu state
        anchorEl: null,
    };

    reset_range(range) {
        const {ranges, pred_ranges, iteration} = this.state;
        let new_ranges = ranges.slice();
        remove(range, new_ranges);
        // remove(range, new_pred_ranges);
        this.setState({
            ranges: new_ranges,
            iteration: iteration + 1,
        });
    }

    clear_ranges() {
        this.setState({
            ranges: [],
        });
        this.checkClick([]);
    }

    exclude_example() {
        this.setState({
            ranges: null,
        });
        this.checkClick(null);
    }

    handleMenuOpen(event) {
        this.setState({anchorEl: event.currentTarget});
    }

    handleMenuClick = option => (event) => {
        switch (option) {
            case "clear":
                this.clear_ranges();
                break;
            case "exclude":
                this.exclude_example();
                break;
            case "include":
                this.clear_ranges();
                break;
            default:
                break;
        }
        this.setState({anchorEl: null});
    }
    
    handleMenuClose() {
        this.setState({anchorEl: null, menu_open: false});
    }

    construct_options() {
        const {anchorEl, ranges} = this.state;
        const inclusion_state = is_valid(ranges) ? "exclude" : "include";
        const options = ["clear", inclusion_state];
        const open = is_valid(anchorEl);
        return (
        <span>
            <IconButton
                aria-label="More"
                aria-owns={open ? 'long-menu' : undefined}
                aria-haspopup="true"
                onClick={this.handleMenuOpen.bind(this)}
            >
            <MoreHorizIcon />
            </IconButton>
            <Menu
                id="long-menu"
                anchorEl={anchorEl}
                open={open}
                onClose={this.handleMenuClose.bind(this)}
                PaperProps={{
                    style: {
                    maxHeight: ITEM_HEIGHT * 4.5,
                    width: 200,
                    },
                }}
            >
            {options.map(option => (
                <MenuItem
                    key={`menu_item_${option}`}
                    onClick={this.handleMenuClick(option).bind(this)}>
                {option}
                </MenuItem>
            ))}
            </Menu>
        </span>
        );
    }

    smartRenderer(lettersNode, range, rangeIndex) {
        if (range.mode === PREDICTION) {
            return underlineRender(
                lettersNode,
                range,
                rangeIndex,
                null,//{onClick: this.reset_range.bind(this, range)},
                "predicted"
            );
        } else if (range.mode === LABEL) {
            return highlightRender(
                lettersNode,
                range,
                rangeIndex,
                {onClick: this.reset_range.bind(this, range)},
                "label",
            );
        } else if (range.mode === COMBINATION) {
            return combinedRender(
                lettersNode,
                range,
                rangeIndex,
                {onClick: this.reset_range.bind(this, range)},
            );
        }
    }

    tooltipRenderer(lettersNode, range, rangeIndex, _) {
        const {classifier_class} = this.props;
        // return (
        // <Tooltip
        //     key={`tool_tip_${range.data.id}-${rangeIndex}`}
        //     title={
        //         <React.Fragment>
        //             {classifier_class}
        //         </React.Fragment>
        //     }
        //     animation="zoom"
        //     >
        //     {this.smartRenderer(
        //         lettersNode,
        //         range,
        //         rangeIndex,
        //     )}
        // </Tooltip>
        // );

        return this.smartRenderer(
            lettersNode,
            range,
            rangeIndex,
        );
    }
  
    customRenderer(currentRenderedNodes, currentRenderedRange, currentRenderedIndex, onMouseOverHighlightedWord) {
        return this.tooltipRenderer(
            currentRenderedNodes,
            currentRenderedRange,
            currentRenderedIndex,
            onMouseOverHighlightedWord,
        );
    }

    onTextHighlighted(range) {
        const {ranges, word_indexes, word_starts, word_ends, iteration} = this.state;
        let start_idx = range.start;
        let end_idx = range.end;

        if (word_indexes[start_idx] === -1) {
            // starts on space, continue to the next word
            while (start_idx < word_indexes.length && word_indexes[start_idx] < 0) {
                start_idx++;
            }
        }

        if (word_indexes[end_idx] === -1) {
            // starts on space, continue to the previous word
            while (end_idx >= 0 && word_indexes[end_idx] < 0) {
                end_idx--;
            }
        }

        const start_word = word_indexes[start_idx];
        const end_word = word_indexes[end_idx];
        range.start = word_starts[start_word];
        range.end = word_ends[end_word];
        range.word_start = start_word;
        range.word_end = end_word;
        range.mode = LABEL;
        range.data.id = `range_start_${range.word_start}_end_${range.word_end}`;

        let res = null;
        if (is_valid_range(range)) {
            const new_ranges = ranges.slice();
            new_ranges.push(range);
            res = highlightRange(range);
            this.setState({
                ranges: new_ranges,
                iteration: iteration + 1,
            });
            this.checkClick(new_ranges);
        }
        window.getSelection().removeAllRanges();
        return res;
    }

    create_map() {
        const {example} = this.props;
        const example_text = example.join(' ');
        const word_idxes = [];
        const word_starts = [];
        const word_ends = [];
        let word_start = -1;
        let word_index = 0;
        for (let i = 0; i < example_text.length; i++) {
            if (example_text[i] === ' ') {
                // since everything is guaranteed to be a single space seperated
                // a space marks the end of a word and the start of the next word

                // at the start of a word
                word_starts.push(word_start + 1);
                word_index++;
                word_start = i;
                if (word_idxes.length > 0 && word_idxes[word_idxes.length - 1] !== -1) {
                    // at the end of a word
                    word_ends.push(word_idxes.length - 1);
                }
                word_idxes.push(-1);
            } else {
                // in the middle of a word, so add the word id (word_index)
                // to the word_idxes list
                word_idxes.push(word_index);
            }
        }

        if (word_start + 1 < example_text.length) {
            word_starts.push(word_start + 1);
        }

        if (word_idxes.length > 0 && word_idxes[word_idxes.length - 1] !== -1) {
            word_ends.push(word_idxes.length - 1);
        }

        this.setState({
            word_indexes: word_idxes,
            word_starts: word_starts,
            word_ends: word_ends,
        });
    }

    componentDidMount() {
        this.create_map();
    }

    componentWillReceiveProps(nextProps) {
        const {example, example_prediction} = nextProps;
        if (is_valid(example_prediction) && example_prediction !== this.props.example_prediction) {
            const pred_ranges = compute_ranges(
                example.join(' '),
                getLast(example_prediction.ranges),
                null,
                {mode: PREDICTION},
            );
            this.setState({pred_ranges});
        }
    }

    checkClick(ranges) {
        const {example, doneLabelFunc, index} = this.props;
        doneLabelFunc(index, example, ranges);
    }

    getShowRanges() {
        const {show_predictions} = this.props;
        let {ranges, pred_ranges} = this.state;
        ranges = is_valid(ranges) ? ranges : [];
        set_array_props(pred_ranges, {mode: PREDICTION});
        set_array_props(ranges, {mode: LABEL});
        let all_ranges = show_predictions ? merge_ranges(ranges, pred_ranges) : ranges;
        all_ranges = all_ranges.filter(x => x);
        return all_ranges;
    }

    render() {
        const {classes, example, index} = this.props;
        const {iteration, ranges} = this.state;
        const show_ranges = this.getShowRanges();
        const cardStyle = {}
        if (!is_valid(ranges)) {
            cardStyle.backgroundColor = grey[400];
        }

        return (
            <Card className={classes.card} >
            <CardContent style={{paddingBottom: 0, ...cardStyle}}>
              <Typography variant="h5" component="h2">
                <CustomHighlightable
                    ranges={show_ranges}
                    enabled={is_valid(ranges)}
                    onTextHighlighted={this.onTextHighlighted.bind(this)}
                    id={`highlight_content_${index}_${iteration}`}
                    text={example.join(' ')}
                    rangeRenderer={this.customRenderer.bind(this)}
                />
              </Typography>
                {this.construct_options()}
            </CardContent>
          </Card>
        );
    }
}

ExampleCard.propTypes = {
    classes: PropTypes.object.isRequired,
    classifier_class: PropTypes.string,
    dataset_id: PropTypes.number,
    example: PropTypes.array,
    example_prediction: PropTypes.object,
    doneLabelFunc: PropTypes.func,
    index: PropTypes.number,
    show_predictions: PropTypes.bool,
};

export default withRoot(withStyles(styles)(ExampleCard));