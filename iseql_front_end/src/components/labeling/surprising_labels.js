import React from 'react';
import PropTypes from 'prop-types';
import { withStyles } from '@material-ui/core/styles';
import ListItem from '@material-ui/core/ListItem';
import Typography from '@material-ui/core/Typography';
import List from '@material-ui/core/List';
import Divider from '@material-ui/core/Divider';
import ListItemText from '@material-ui/core/ListItemText';
import blueGrey from '@material-ui/core/colors/blueGrey';
import Tooltip from '@material-ui/core/Tooltip';
import IconButton from '@material-ui/core/IconButton';
import Menu from '@material-ui/core/Menu';
import MenuItem from '@material-ui/core/MenuItem';
import MoreVertIcon from '@material-ui/icons/MoreVert';
import ArrowUpward from '@material-ui/icons/ArrowUpward';
import ArrowDownward from '@material-ui/icons/ArrowDownward';
// import NewReleasesOutlined from '@material-ui/icons/NewReleasesOutlined';
import Select from '@material-ui/core/Select';
import FiberNewOutlined from '@material-ui/icons/FiberNewOutlined'
import StarRate from '@material-ui/icons/StarRate';
import withRoot from '../../withRoot';
import red from '@material-ui/core/colors/red';
import green from '@material-ui/core/colors/green'
import purple from '@material-ui/core/colors/purple';
import MultiLineGraphVis from '../../visualizations/core/vega_lite_multi_line_graph';
import SortableTable from '../core/sortable_table';
import {getLast, is_valid, get_index, get_items, get_percentage} from '../../utils';

const NewReleasesOutlined = FiberNewOutlined;

const SHADE = 700;

const styles = theme => ({
root: {
    textAlign: 'center',
    paddingTop: theme.spacing.unit * 20,
},
list: {
    backgroundColor: theme.palette.background.paper,
    marginLeft: theme.spacing.unit * 1,
    marginRight: theme.spacing.unit * 1,
    overflow: "scroll",
}
});

const ITEM_HEIGHT = 48;

const OPTIONS = [
    "Predicted Entities",
    "Labeled Entities",
    "Discovered Entities",
];

class SurprsingLabels extends React.Component {
    state = {
        selected_menu_option: OPTIONS[0],
        menu_open: false,
        anchorEl: null,
    }

    surprising_words(prediction_data) {
        const ent_count = {};
        const label_count = {};
        const prev_ent_count = {};
        for (const i in prediction_data) {
            const pred_data = prediction_data[i][1];
            if (!is_valid(pred_data)) {
                break;
            }
            const entities = getLast(pred_data.entities);
            const prev_entities = get_index(pred_data.entities, pred_data.entities.length - 2);
            const label_entities = pred_data.real_entities;
            for (const ei in entities) {
                const ent_data = entities[ei];
                if (!(ent_data in ent_count)) {
                    ent_count[ent_data] = {count: 0, prev_count: 0, discovered: true};
                }
                ent_count[ent_data].count++;
            }

            if (is_valid(prev_entities)) {
                for (const ei in prev_entities) {
                    const ent_data = prev_entities[ei];
                    if (!(ent_data in prev_ent_count)) {
                        prev_ent_count[ent_data] = {count: 0, prev_count: 0};
                    }
                    prev_ent_count[ent_data].count++;
                }
            }

            for (const li in label_entities) {
                const label_ent_data = label_entities[li];
                if (!(label_ent_data in label_count)) {
                    label_count[label_ent_data] = {count: 0, prev_count: 0};
                }
                label_count[label_ent_data].count++;
                label_count[label_ent_data].prev_count++;
            }
        }

        const discovered_count = Object.assign({}, ent_count);
        for (const label_ent in label_count) {
            if (label_ent in discovered_count) {
                delete discovered_count[label_ent];
                ent_count[label_ent].discovered = false;
            }
        }


        const prev_res = get_items(prev_ent_count);
        prev_res.sort(function(a, b){return b[1].count - a[1].count;});

        for (let i = 0; i < prev_res.length; i++ ){ 
            const rank = i + 1;
            const ent_data = prev_res[i];
            const ent_name = ent_data[0];
            const ent_dat_count = ent_data[1].count;
            if (ent_name in ent_count) {
                ent_count[ent_name].prev_rank = rank;
                ent_count[ent_name].prev_count = ent_dat_count;
            }
        }

        const ent_res = get_items(ent_count);
        ent_res.sort(function(a, b){return b[1].count - a[1].count;});
        const label_res = get_items(label_count);
        label_res.sort(function(a, b){return b[1].count - a[1].count;});
        const discovered_res = get_items(discovered_count);
        discovered_res.sort(function(a, b) {return b[1].count - a[1].count;});

        return {ent_res, label_res, discovered_res};
    }

    handleMenuOpen(event) {
        //console.log(event.currentTarget);
        //this.setState({selected_menu_option: event.currentTarget, menu_open: true});
        this.setState({anchorEl: event.currentTarget, menu_open: true});
    }

    handleMenuClick = option => (event) => {
        this.setState({selected_menu_option: option, anchorEl: null, menu_open: true});
    }
    
    handleMenuClose() {
        this.setState({anchorEl: null, menu_open: false});
    }

    componentDidMount() {
        const {predictions} = this.props;
        if (is_valid(predictions)) {
            let {ent_res, label_res, discovered_res} = this.surprising_words(predictions);
            this.props.updateTopEntities(ent_res.map(ent_data => ent_data[0]).slice(0, 3));
            this.setState({ent_res, label_res, discovered_res});
        }
    }

    componentWillReceiveProps(newProps) {
        const {predictions} = newProps;
        if (is_valid(predictions) && JSON.stringify(predictions) !== JSON.stringify(this.props.predictions)) {
            let {ent_res, label_res, discovered_res} = this.surprising_words(predictions);
            this.props.updateTopEntities(ent_res.map(ent_data => ent_data[0]).slice(0, 3));
            this.setState({ent_res, label_res, discovered_res});
        }
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
            
          {OPTIONS.map(option => (
            <MenuItem
                key={option}
                selected={option === selected_menu_option}
                onClick={this.handleMenuClick(option).bind(this)}>
              {option}
            </MenuItem>
          ))}
        </Menu>
        </span>
        );
    }

    get_icon(delta, fontSize) {
        // delta > 0 ?
        //                 <ArrowUpward style={{color: green[SHADE], fontSize}} /> :
        //                 (delta === 0 ? null : <ArrowDownward style={{color: red[SHADE], fontSize}} />);
        if (isNaN(delta)) {
            return this.new_rank_icon(fontSize)
        }
        if (delta > 0) {
            return this.increase_rank_icon(fontSize);
        } else if (delta < 0) {
            return this.decrease_rank_icon(fontSize);
        } else if (delta === 0) {
            return null;
        }
    }

    new_rank_icon(fontSize) {
        return <Tooltip
            title={
                <React.Fragment>
                    No previous rank
                </React.Fragment>
            }
            animation="zoom"
            >
            <NewReleasesOutlined style={{fontSize: fontSize + 5}}/>
        </Tooltip>;
    }

    increase_rank_icon(fontSize) {
        return <Tooltip
            title={
                <React.Fragment>
                    Went up in rank
                </React.Fragment>
            }
            animation="zoom"
            >
            <ArrowUpward style={{color: green[SHADE], fontSize}} />
        </Tooltip>;
    }

    decrease_rank_icon(fontSize) {
        return <Tooltip
            title={
                <React.Fragment>
                    Went down in rank
                </React.Fragment>
            }
            animation="zoom"
            >
            <ArrowDownward style={{color: red[SHADE], fontSize}} />
        </Tooltip>;
    }

    discovered_icon(fontSize) {
        return <Tooltip
            title={
                <React.Fragment>
                    Discovered (not labeled)
                </React.Fragment>
            }
            animation="zoom"
            >
           <StarRate style={{fontSize}} />
        </Tooltip>;
    }

    render() {
        const {classes, predictions, height, updateTopEntities} = this.props;
        if (!is_valid(predictions)) {
            return null;
        }
        const {selected_menu_option, ent_res, label_res, discovered_res} = this.state;

        let res = [];
        switch(selected_menu_option) {
            case OPTIONS[0]:
                res = ent_res;
                break;
            case OPTIONS[1]:
                res = label_res;
                break;
            case OPTIONS[2]:
                res = discovered_res;
                break;
            default:
                break;
        }

        if (!is_valid(res)) {
            res = [];
        }

        res = res.filter(x => x[1].count > 0);

        const max_count = is_valid(res) && res.length >= 0 && is_valid(res[0]) ? res[0][1].count : 0;

        res = res.slice(0, res.length >= 100 ? 100 : res.length);

        return (
            <div>
                <Typography variant="h6" gutterBottom style={{borderBottom: purple[100], margin: 0}}>
                    {selected_menu_option}
                    {this.construct_menu()}
                </Typography>
                {max_count === 0 ? <Typography variant="subtitle2" gutterBottom style={{top: "50%", textAlign: "center"}}>
                    No Results
                </Typography>:
                <List className={classes.list} style={{height, padding: 0, paddingRight: 10}} dense={true}>
                    {res.map((ent_data, index) => {
                    
                    // const delta = ent_data[1].count - ent_data[1].prev_count;
                    let delta = (index + 1) - ent_data[1].prev_rank;
                    // delta = is_valid(delta) && !isNaN(delta) ? delta : Number.MIN_VALUE;
                    delta = -delta;
                    const fontSize = 16;
                    const rank_icon = this.get_icon(delta, fontSize);
                    const star_icon = ent_data[1].discovered ? this.discovered_icon(fontSize) : null;
                    return (
                    <Tooltip title={
                        <React.Fragment>
                            {
                                <span>
                                    <div>{ent_data[0]}</div>
                                    <div>Count: {ent_data[1].count}</div>
                                    <div>Previous Count: {ent_data[1].prev_count}</div>
                                </span>
                            }
                        </React.Fragment>
                    } placement="left">
                    <div>
                        <ListItem style={{padding: 0}}>
                        <div
                            key={`div_${ent_data[0]}_${ent_data[1].count}`}
                            style={{
                                background: `-webkit-linear-gradient(left, ${blueGrey[200]} ${get_percentage(ent_data[1].count, max_count)}%, white 0%)`,
                                display: 'inline-block',
                                flexGrow: 1,
                                padding: 0,
                        }}>
                            <ListItemText
                                primary={
                                    <React.Fragment>
                                        <Typography component="span" style={{fontSize}} color="textPrimary">
                                            {`${ent_data[0]}`}
                                            {star_icon}
                                        </Typography>
                                    </React.Fragment>
                                }
                            />
                        </div>
                        {rank_icon}
                        </ListItem>
                        <Divider />
                    </div>
                    </Tooltip>
                    );})}
                </List>
                }
            </div>
        );
    }
}

SurprsingLabels.propTypes = {
    data: PropTypes.object,
    predictions: PropTypes.array,
    height: PropTypes.number,
    updateTopEntities: PropTypes.func,
};

export default withRoot(withStyles(styles)(SurprsingLabels));