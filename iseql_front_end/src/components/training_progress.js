import React from 'react';
import PropTypes from 'prop-types';
import { withStyles } from '@material-ui/core/styles';
import withRoot from '../withRoot';
import LineGraphVis from '../visualizations/core/vega_lite_line_graph';
import MultiLineGraphVis from '../visualizations/core/vega_lite_multi_line_graph';
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
visualization : {
    display: "inline-block",
    margin: theme.spacing.unit * 2
}
});

class TrainingProgress extends React.Component {
    render() {
        const {classifier_class, training_progress_data} = this.props;
        return (
            <div>
                {/* <LineGraphVis
                    x_field={"epoch_number"}
                    y_field={"train_f1_avg"}
                    x_type={"quantitative"}
                    y_type={"quantitative"}
                    title={"Training F1 Average"}
                    description={"Training F1 Average"}
                    data={training_progress_data}
                />
                <LineGraphVis
                    x_field={"epoch_number"}
                    y_field={"train_accuracy"}
                    x_type={"quantitative"}
                    y_type={"quantitative"}
                    title={"Training Accuracy"}
                    description={"Training Accuracy"}
                    data={training_progress_data}
                />
                <LineGraphVis
                    x_field={"epoch_number"}
                    y_field={"train_loss_avg"}
                    x_type={"quantitative"}
                    y_type={"quantitative"}
                    title={"Training Loss Average"}
                    description={"Training Loss Average"}
                    data={training_progress_data}
                /> */}
                <div>
                    {/* <LineGraphVis
                        x_field={"epoch_number"}
                        y_field={`train_f1.${classifier_class}.f1`}
                        x_type={"quantitative"}
                        y_type={"quantitative"}
                        title={`Training ${classifier_class} F1`}
                        description={`Training ${classifier_class} F1`}
                        data={training_progress_data}
                    />
                    <LineGraphVis
                        x_field={"epoch_number"}
                        y_field={`train_f1.${classifier_class}.recall`}
                        x_type={"quantitative"}
                        y_type={"quantitative"}
                        title={`Training ${classifier_class} Recall`}
                        description={`Training ${classifier_class} Recall`}
                        data={training_progress_data}
                    />
                    <LineGraphVis
                        x_field={"epoch_number"}
                        y_field={`train_f1.${classifier_class}.precision`}
                        x_type={"quantitative"}
                        y_type={"quantitative"}
                        title={`Training ${classifier_class} precision`}
                        description={`Training ${classifier_class} precision`}
                        data={training_progress_data}
                    /> */}
                    <MultiLineGraphVis
                        x_field={["epoch_number", "epoch_number", "epoch_number"]}
                        y_field={[
                            `train_f1.${classifier_class}.f1`,
                            `train_f1.${classifier_class}.precision`,
                            `train_f1.${classifier_class}.recall`,
                        ]}
                        y_title={null}
                        x_type={["quantitative", "quantitative", "quantitative"]}
                        y_type={["quantitative", "quantitative", "quantitative"]}
                        title={`Training ${classifier_class} precision`}
                        description={`Training ${classifier_class} precision`}
                        data={training_progress_data}
                    />
                    <MultiLineGraphVis
                        x_field={["epoch_number", "epoch_number", "epoch_number"]}
                        y_field={[
                            `valid_f1.${classifier_class}.f1`,
                            `valid_f1.${classifier_class}.precision`,
                            `valid_f1.${classifier_class}.recall`,
                        ]}
                        y_title={null}
                        x_type={["quantitative", "quantitative", "quantitative"]}
                        y_type={["quantitative", "quantitative", "quantitative"]}
                        title={`Training ${classifier_class} precision`}
                        description={`Training ${classifier_class} precision`}
                        data={training_progress_data}
                    />
                </div>
            </div>
        );
    }
}

TrainingProgress.propTypes = {
    data: PropTypes.object,
    classifier_class: PropTypes.string,
    training_progress_data: PropTypes.array,
};

export default withRoot(withStyles(styles)(TrainingProgress));