import React from 'react';
import PropTypes from 'prop-types';
import Vega from 'react-vega';
import { withStyles } from '@material-ui/core/styles';
import withRoot from '../withRoot';
import {construct_spec} from './word_k_chart_spec';
// import {construct_spec} from './topk';

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
});

class WordKVis extends React.Component {

    prepareData() {
        const {data} = this.props;
        const res = []
        for (const i in data) {
            const ent = data[i];
            res.push(ent);
        }

        return res.join(" ");
    }

    render() {
        const data = this.prepareData();
        const spec = construct_spec(data);
        return <Vega spec={spec}/>;
    }
}

WordKVis.propTypes = {
    classes: PropTypes.object.isRequired,
    classifier_class: PropTypes.string,
    dataset_id: PropTypes.number,
    data: PropTypes.string, // an array of entities
};

export default withRoot(withStyles(styles)(WordKVis));