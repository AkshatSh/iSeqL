import React from 'react';
import PropTypes from 'prop-types';
import { withStyles } from '@material-ui/core/styles';
import IconButton from '@material-ui/core/IconButton';
import Menu from '@material-ui/core/Menu';
import MenuItem from '@material-ui/core/MenuItem';
import MoreVertIcon from '@material-ui/icons/MoreVert';
import Select from '@material-ui/core/Select';
import withRoot from '../../withRoot';
import MultiLineGraphVis from '../../visualizations/core/vega_lite_multi_line_graph';
import {is_valid, prettyPrint} from '../../utils';

const styles = theme => ({
root: {
    textAlign: 'center',
    paddingTop: theme.spacing.unit * 20,
},
container: {
    display: 'flex',
},
vis: {
    flexGrow: 9,
},
menu: {
    flexGrow: 1,
    paddingRight: 10,
}
});


const ITEM_HEIGHT = 48;

const METRICS = [
    'f1',
    'precision',
    'recall',
];

class TrainSummary extends React.Component {
    state = {
        selected_menu_option: METRICS[0],
        menu_open: false,
        anchorEl: null,
    }

    handleMenuOpen(event) {
        this.setState({anchorEl: event.currentTarget, menu_open: true});
    }

    handleMenuClick = option => (event) => {
        this.setState({selected_menu_option: option, anchorEl: null, menu_open: true});
    }
    
    handleMenuClose() {
        this.setState({anchorEl: null, menu_open: false});
    }

    construct_menu() {
        const {selected_menu_option,anchorEl, menu_open} = this.state;
        const open = is_valid(anchorEl);
        return (<span>
        <IconButton
          aria-label="More"
          aria-owns={open ? 'long-menu' : undefined}
          aria-haspopup="true"
          onClick={this.handleMenuOpen.bind(this)}
          style={{padding: 8}}
        >
          <MoreVertIcon />
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
          {METRICS.map(option => (
            <MenuItem
                key={option}
                selected={option === selected_menu_option}
                onClick={this.handleMenuClick(option).bind(this)}>
              {prettyPrint(option)}
            </MenuItem>
          ))}
        </Menu>
        </span>
        );
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
              Quality Score: {prettyPrint(option)}
            </MenuItem>
          ))}
      </Select>;
    }

    create_data(labeled_set_sizes, train_summary) {
        const data = []
        for (var i in train_summary) {
            const train_data = train_summary[i];
            const labeled_set_size = labeled_set_sizes[i];
            data.push({
                labeled_set_size,
                train_metric: train_data.train_f1.ADR[this.state.selected_menu_option],
                test_metric: train_data.valid_f1.ADR[this.state.selected_menu_option],
            });
        }

        return data;
    }
    render() {
        const {labeled_set_sizes, train_summary} = this.props;
        if (!is_valid(labeled_set_sizes) || !is_valid(train_summary)) {
            return null;
        }

        const data = this.create_data(labeled_set_sizes, train_summary);

        return (
            <div>
                {this.construct_selector()}
                <MultiLineGraphVis
                    layer_titles={["Train Quality Score", "Test Quality Score"]}
                    x_field={["labeled_set_size", "labeled_set_size"]}
                    y_field={[
                        'train_metric',
                        'test_metric',
                    ]}
                    chart_type={"line"}
                    x_type={["quantitative", "quantitative"]}
                    x_title={[null, "# Labeled Examples"]}
                    y_type={["quantitative", "quantitative"]}
                    y_title={null}
                    // title={`${prettyPrint(this.state.selected_menu_option)}`}
                    description={`Summarizes the F1 scores on Train and Devsets`}
                    data={data}
                />
            </div>
        );
    }
}

TrainSummary.propTypes = {
    data: PropTypes.object,
    train_summary: PropTypes.array,
    labeled_set_sizes: PropTypes.array,
};

export default withRoot(withStyles(styles)(TrainSummary));