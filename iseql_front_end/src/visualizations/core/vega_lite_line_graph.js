
import React from 'react';
import PropTypes from 'prop-types';
import VegaLite from 'react-vega-lite';
import { withStyles } from '@material-ui/core/styles';
import withRoot from '../../withRoot';

export function construct_spec(
    x_field,
    x_type,
    y_field,
    y_type,
    description="Description",
) {
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v3.json",
        "description": description,
        "mark": {
            "type": "line",
            "point": true
        },
        "encoding": {
          "x": {"field": x_field, "type": x_type},
          "y": {"field": y_field, "type": y_type}
        }
    };
}

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
    vis: {
        display: 'inline-block',
        margin: theme.spacing.unit,
    }
});

class LineGraphVis extends React.Component {
    render() {
        const {classes, x_field, y_field, x_type, y_type, description, data} = this.props;
        const spec = construct_spec(x_field, x_type, y_field, y_type, description);
        return <VegaLite className={classes.vis} spec={spec} data={{
            values: data,
        }}/>;
    }
}

LineGraphVis.propTypes = {
    classes: PropTypes.object.isRequired,
    x_field: PropTypes.string,
    x_type: PropTypes.string,
    y_field: PropTypes.string,
    y_type: PropTypes.string,
    title: PropTypes.string,
    description: PropTypes.string,
    data: PropTypes.array,
};

export default withRoot(withStyles(styles)(LineGraphVis));