import React from 'react';
import PropTypes from 'prop-types';
import { withStyles } from '@material-ui/core/styles';
import MenuItem from '@material-ui/core/MenuItem';
import Select from '@material-ui/core/Select';
import withRoot from '../../withRoot';
import MultiLineGraphVis from '../../visualizations/core/vega_lite_multi_line_graph';
import {is_valid, prettyPrint} from '../../utils';

const styles = theme => ({
    root: {
        textAlign: 'center',
        paddingTop: theme.spacing.unit * 20,
    },
});

const ITEM_HEIGHT = 48;
const METRICS = [
    'total_flipped',
    'pos_flipped',
    'neg_flipped',
];

const PRETTY_NAME = {
    'total_flipped': 'Total Flipped',
    'pos_flipped': 'Positive Flipped',
    'neg_flipped': 'Negative Flipped',
}

class LabelProgress extends React.Component {
    state = {
        selected_menu_option: METRICS[0],
        menu_open: false,
        anchorEl: null,
        selecotr_open: false,
    }

    handleChange(event) {
        this.setState({selected_menu_option: event.target.value});
    }
  
    handleClose() {
        this.setState({
            selector_open: false,
        });
    }

    handleOpen() {
        this.setState({
            selector_open: true,
        });
    }

    construct_selector() {
        const {selector_open, selected_menu_option} = this.state;

        return <Select
            style={{width: 200, fontSize: 12, height: 20}}
            open={selector_open}
            onClose={this.handleClose.bind(this)}
            onOpen={this.handleOpen.bind(this)}
            value={selected_menu_option}
            onChange={this.handleChange.bind(this)}
            inputProps={{
                name: 'metric_value',
                id: 'demo-controlled-open-select-metric_val',
            }}
        >
        {METRICS.map(option => (
            <MenuItem value={option}>
              {PRETTY_NAME[option]}
            </MenuItem>
          ))}
      </Select>;
    }

    render() {
        const {flipped_data} = this.props;
        const {selected_menu_option} = this.state
        if (!is_valid(flipped_data)) {
            return null;
        }

        return (
            <div>
                {this.construct_selector()}
                <MultiLineGraphVis
                    x_field={["labeled_set_sizes"]}
                    y_field={[
                        selected_menu_option,
                    ]}
                    x_type={["quantitative"]}
                    x_title={[null, "# Labeled Examples"]}
                    chart_type={"area"}
                    y_type={["quantitative"]}
                    y_title={null}
                    title={null}
                    // title={`Flipping Labels: ${PRETTY_NAME[selected_menu_option]}`}
                    layer_titles={["Number of labels flipped"]}
                    description={`Number of labels flipped`}
                    data={flipped_data}
                />
            </div>
        );
    }
}

LabelProgress.propTypes = {
    data: PropTypes.object,
    flipped_data: PropTypes.array,
};

export default withRoot(withStyles(styles)(LabelProgress));