import React from 'react';
import PropTypes from 'prop-types';
import Vega from 'react-vega';
import { withStyles } from '@material-ui/core/styles';
import withRoot from '../withRoot';
import {construct_spec} from './edge_bundling_spec';

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

class EdgeBundleVis extends React.Component {
    render() {
        let {data} = this.props;
        let deps = null;
        ({data, deps} = data);
        const spec = construct_spec(data, deps);
        console.log(JSON.stringify(spec));
        return <Vega spec={spec}/>;
    }
}

EdgeBundleVis.propTypes = {
    classes: PropTypes.object.isRequired,
    classifier_class: PropTypes.string,
    dataset_id: PropTypes.number,
    data: PropTypes.object, // an array of entities
};

export default withRoot(withStyles(styles)(EdgeBundleVis));