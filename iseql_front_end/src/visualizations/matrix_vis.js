import React from 'react';
import PropTypes from 'prop-types';
import Vega from 'react-vega';
import { withStyles } from '@material-ui/core/styles';
import withRoot from '../withRoot';
import {construct_spec} from './matrix_graph_spec';

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

class MatrixVis extends React.Component {
    render() {
        const {data, ent_type} = this.props;
        const spec = construct_spec({
            "text": {"signal": `'Co-occurence of Top '+k+' ${ent_type} Entities'`}
          });
        return <Vega spec={spec} data={data}/>;
    }
}

MatrixVis.propTypes = {
    classes: PropTypes.object.isRequired,
    classifier_class: PropTypes.string,
    ent_type: PropTypes.string,
    dataset_id: PropTypes.number,
    data: PropTypes.object, // an array of entities
};

export default withRoot(withStyles(styles)(MatrixVis));