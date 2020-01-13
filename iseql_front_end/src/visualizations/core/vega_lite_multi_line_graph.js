
import React from 'react';
import PropTypes from 'prop-types';
import VegaLite from 'react-vega-lite';
import Vega from 'react-vega';
import { Handler } from 'vega-tooltip';
import Typography from '@material-ui/core/Typography';
import lightBlue from '@material-ui/core/colors/lightBlue';
import orange from '@material-ui/core/colors/orange';
import green from '@material-ui/core/colors/green'
import { withStyles } from '@material-ui/core/styles';
import { key_by_string } from '../../utils';
import withRoot from '../../withRoot';

const SHADE = 700;
export const COLOR_SCHEME = [orange[SHADE], lightBlue[SHADE], green[SHADE]];

// export const COLOR_SCHEME = [lightBlue[SHADE], lightBlue[SHADE], lightBlue[SHADE]];

function simple_spec(
    chart_type,
    x_fields,
    x_titles,
    x_types,
    y_fields,
    y_types,
    layer_titles,
    description="Description",
    height=20,
) {
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v3.json",
        "description": description,
        // "title": layer_title,
        "width": 250,
        "height": height * 5,
        "mark": {
            "type": chart_type,
            "line": true,
            "point": true,
        },
        "encoding": {
            "x": {"field": "x", "type": "quantitative", "axis": {"title": null},
            "scale": {
                "zero": false
            }
        },
            "y": {"field": "value", "type": "quantitative", "axis": {"labels": true, "title": null}, 
        },
        "color": {"field": "type", "type": "nominal"}
        },
    };
}

export function construct_spec(
    title,
    chart_type,
    x_fields,
    x_titles,
    x_types,
    y_fields,
    y_title,
    y_types,
    layer_titles,
    description="Description",
    height=20,
) {
    const layers = [];
    const combine_key = chart_type === "area"  ? "layer" : "layer";
    for (let i = 0; i < x_fields.length; i++) {
        const x_field = x_fields[i];
        const y_field = y_fields[i];
        const x_type = x_types[i];
        const y_type = y_types[i];
        const x_title = x_titles[i];
        const layer_title = layer_titles[i];
        const color = COLOR_SCHEME[i];
        layers.push({
            "width": 250,
            "height": height * (chart_type === "line" ? 5 : 5),
            "mark": {
                "type": chart_type,
                "line": true,
                "point": true,
            },
            "encoding": {
              "x": {"field": x_field, "type": x_type, "axis": {"title": x_title},
              "scale": {
                "zero": false
                }
            },
              "y": {"field": y_field, "type": y_type, "axis": {"labels": true, "title": y_title}, 
            //   "scale": {
            //     "zero": false
            //   }
            },
               "color": {"value": color},
            },
        });
    }
    const res = {
        "$schema": "https://vega.github.io/schema/vega-lite/v3.json",
        "description": description,
        "title": title,
    };

    if (chart_type === "area") {
        res["resolve"] = {"scale": {"y": "independent"}};
    }

    res[combine_key] = layers;

    return res;
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
        // margin: theme.spacing.unit,
    },
    colorBox: {
        width: '10px',
        height: '10px',
        display: 'inline-block',
        marginRight: theme.spacing.unit,
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
        marginRight: theme.spacing.unit,
        fontSize: 12,
    },
    legend: {
        marginRight: theme.spacing.unit,
    }
});

function simplify_data(x_field, y_titles, y_fields, data) {
    // given raw data and a series of x fields, and a y field
    const vega_data = [];
    for (let i = 0; i < data.length; i++) {
        const curr_data = data[i];
        for (let iy = 0; iy < y_fields.length; iy++) {
            const y_field = y_fields[iy];
            const y_title = y_titles[iy];
            vega_data.push(
                {
                    value: key_by_string(curr_data, y_field),
                    type: y_title,
                    x: key_by_string(curr_data, x_field),
                }
            )
        }
    }

    return vega_data;

}


class MultiLineGraphVis extends React.Component {

    create_legend() {
        const {classes, y_field, layer_titles} = this.props;

        const labels = [];
        for (let i = 0; i < y_field.length; i++) {
            const color = COLOR_SCHEME[i];
            const label_title = layer_titles[i];
            labels.push(
                <span>
                    <div className={classes.colorBox} style={{backgroundColor: color}}></div>
                    <Typography variant="body2" className={classes.label}>
                        {label_title}
                    </Typography>
                </span>
            );
        }

        return (
        <div className={classes.legend}>
            {labels}
        </div>
        )
    }
    render() {
        const {chart_type, title, classes, layer_titles, x_field, x_title, y_field, x_type, y_type, y_title, description, data} = this.props;
        const spec = construct_spec(title, chart_type, x_field, x_title, x_type, y_field, y_title, y_type,layer_titles, description);
        // const t_simple_spec = simple_spec(chart_type, x_field, x_title, x_type, y_field, y_type,layer_titles, description);
        // // console.log(JSON.stringify(spec));
        // // console.log(JSON.stringify(data));
        // if (chart_type === "line") {
        //     const simple_data = simplify_data(x_field[0], layer_titles, y_field, data);
        //     return <VegaLite className={classes.vis} spec={t_simple_spec} data={{
        //         values: simple_data,
        //     }}/>
        // }
        return <div>
            <VegaLite className={classes.vis} spec={spec} tooltip={new Handler().call} data={{
                values: data,
            }}/>
            {this.create_legend()}
        </div>;
    }
}

MultiLineGraphVis.propTypes = {
    classes: PropTypes.object.isRequired,
    x_field: PropTypes.array,
    x_title: PropTypes.array,
    x_type: PropTypes.array,
    y_field: PropTypes.array,
    y_title: PropTypes.array,
    y_type: PropTypes.array,
    title: PropTypes.string,
    layer_titles: PropTypes.array,
    description: PropTypes.string,
    data: PropTypes.array,
    chart_type: PropTypes.string,
};

export default withRoot(withStyles(styles)(MultiLineGraphVis));