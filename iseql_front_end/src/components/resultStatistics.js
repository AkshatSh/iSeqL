import React from 'react';
import PropTypes from 'prop-types';
import { withStyles } from '@material-ui/core/styles';
import withRoot from '../withRoot';
import EntKVis from '../visualizations/entk_vis';
import WordKVis from '../visualizations/wordk_vis';
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

class ResultStatistics extends React.Component {

    construct_word_dataset() {
        const {data} = this.props;
        const label_word_data = [];
        const entity_word_data = [];
        for (const s_id in data) {
            const entry_data = data[s_id];
            const sentence_data = entry_data[1];
            const entities = sentence_data.entities;
            const real_entities = sentence_data.real_entities;

            for (const i in entities) {
                entity_word_data.push(
                    entities[i]
                );
            }

            for (const i in real_entities) {
                label_word_data.push(
                    real_entities[i]
                );
            }
        }

        return {entity_word_data, label_word_data};
    }

    render() {
        const {classes, dataset_id, classifier_class, show_labels, show_predictions} = this.props;
        let entity_word_data, label_word_data = null;
        ({entity_word_data, label_word_data} = this.construct_word_dataset());
        const construct_vis = (data, title) => {
            return  (
                <div>
                    <Typography variant="h5">
                        {title}
                    </Typography>
                    <div className={classes.visualization}>
                        <EntKVis
                            dataset_id={dataset_id}
                            classifier_class={classifier_class}
                            data={data}
                        />
                    </div>
                    <div className={classes.visualization}>
                        <WordKVis
                            dataset_id={dataset_id}
                            classifier_class={classifier_class}
                            data={data}
                        />
                    </div>
                </div>
            );
        }

        return <div>
            {show_predictions ? construct_vis(entity_word_data, "Predicted") : null}
            {show_labels ? construct_vis(label_word_data, "Labeled") : null}
        </div>
    }
}

ResultStatistics.propTypes = {
    classes: PropTypes.object.isRequired,
    classifier_class: PropTypes.string,
    dataset_id: PropTypes.number,
    data: PropTypes.object,
    show_predictions: PropTypes.bool,
    show_labels: PropTypes.bool,
};

export default withRoot(withStyles(styles)(ResultStatistics));