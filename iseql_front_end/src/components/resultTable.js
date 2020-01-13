import React from 'react';
import PropTypes from 'prop-types';
import { withStyles } from '@material-ui/core/styles';
import Paper from '@material-ui/core/Paper';
import TableHighlightCell from './labeling/table_highlight_cell';
import MuiVirtualizedTable from './MuiVirtualizedTable';
import withRoot from '../withRoot';
import {is_valid, set_array_props} from '../utils';
import {highlightableText, combinedHighlightText} from '../utils/highlight_utils';
import { Typography } from '@material-ui/core';

const styles = theme => ({
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
formControl: {
    margin: theme.spacing.unit * 3,
},
predColorBox: {
    width: '10px',
    height: '10px',
    display: 'inline-block',
    backgroundColor: '#e1bee7',
},
labeledColorBox: {
    width: '10px',
    height: '10px',
    display: 'inline-block',
    backgroundColor: '#ffcc80',
},
combinedColorBox: {
    width: '10px',
    height: '10px',
    display: 'inline-block',
    backgroundColor: '#c8e6c9',
},
label : {
    display: 'inline-block',
    margin: theme.spacing.unit * 2,
},
legend: {
    margin: theme.spacing.unit * 2,
}
});

class ResultTable extends React.Component {
    state = {
        predictions: {},
        rows: [],
    };

    prepareRows() {
        const {data} = this.props;
        // const {train, test, unlabeled} = this.state;
        const rows = []
        for (const s_id in data) {
            const id = parseInt(s_id);
            const entry_data = data[s_id];
            const sentence = entry_data[0];
            const sentence_data = entry_data[1];

            const ranges = sentence_data.ranges;
            const entities = sentence_data.entities;
            const real_ranges = sentence_data.real_ranges;
            const real_entities = sentence_data.real_entities;
            const sentence_text = sentence.join(' ');
            const real_entities_text = real_entities.join(', ');
            const entities_text = entities.join(', ');
            rows.push(
                {
                    id,
                    s_id,
                    sentence,
                    ranges,
                    entities,
                    real_ranges,
                    real_entities,
                    sentence_text,
                    real_entities_text,
                    entities_text,
                }
            )
        }

        return rows;
    }

    highlightCellRenderer({cellData, rowData}) {
        const {show_predicted, show_labeled} = this.props;
        const id = rowData.s_id;
        const example_text = rowData.sentence_text;
        return combinedHighlightText(
            example_text,
            show_labeled ? rowData.real_ranges: [],
            show_predicted ? rowData.ranges : [],
            false,
            id,
        );
    }

    prepareColDescriptions() {
        return [
            {
                width: 120,
                label: 'Entry ID',
                dataKey: 's_id',
                numeric: true,
                cellContentRenderer: ({cellData, rowData}) => cellData,
            },
            {
                width: 200,
                flexGrow: 1.0,
                label: 'Sentence',
                dataKey: 'sentence_text',
                cellContentRenderer: this.highlightCellRenderer.bind(this),
            },
            // {
            //     width: 120,
            //     flexGrow: 0.5,
            //     label: 'Predicted Entities',
            //     dataKey: 'entities_text',
            //     cellContentRenderer: ({cellData, rowData}) => cellData,
            // },
            // {
            //     width: 120,
            //     flexGrow: 0.5,
            //     label: 'Labeled Entities',
            //     dataKey: 'real_entities_text',
            //     cellContentRenderer: ({cellData, rowData}) => cellData,
            // },
        ];
    }

    handleChange = name => event => {
        this.setState({ ...this.state, [name]: event.target.checked });
    };

    render() {
        const {classes, include_legend} = this.props;
        const rows = this.prepareRows();
        const column_info = this.prepareColDescriptions();
        return (
            <div>
                {is_valid(include_legend) ?
                    <div className={classes.legend}>
                        <div className={classes.combinedColorBox}></div>
                        <Typography className={classes.label}>Both Predicted and Labeled Entities</Typography>
                        <div className={classes.predColorBox}></div>
                        <Typography className={classes.label}>Predicted Entities</Typography>
                        <div className={classes.labeledColorBox}></div>
                        <Typography className={classes.label}>Labeled Entities</Typography>
                    </div>
                : null}
                <Paper style={{ height: 550, width: '100%' }}>
                    <MuiVirtualizedTable
                        key={"MUIResultTable_rows_length_" + rows.length}
                        rowCount={rows.length}
                        rowGetter={({ index }) => {
                            return rows[index];
                        }}
                        // onRowClick={event => console.log(event)}
                        columns={column_info}
                        rowHeight={({index}) => rows[index].sentence.length * 1.5 + 20}
                    />
                </Paper>
            </div>
        );
    }
}

ResultTable.propTypes = {
    classes: PropTypes.object.isRequired,
    classifier_class: PropTypes.string,
    dataset_id: PropTypes.number,
    data: PropTypes.object,
    is_predicted: PropTypes.bool,
    show_predicted: PropTypes.bool,
    show_labeled: PropTypes.bool,
    include_legend: PropTypes.bool,
};

export default withRoot(withStyles(styles)(ResultTable));