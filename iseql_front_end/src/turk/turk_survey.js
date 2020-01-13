import React from 'react';
import PropTypes from 'prop-types';
import Typography from '@material-ui/core/Typography';
import Button from '@material-ui/core/Button';
import Radio from '@material-ui/core/Radio';
import TextField from '@material-ui/core/TextField';
import FormLabel from '@material-ui/core/FormLabel';
import RadioGroup from '@material-ui/core/RadioGroup';
import FormControlLabel from '@material-ui/core/FormControlLabel';
import FormControl from '@material-ui/core/FormControl';
import CircularProgress from '@material-ui/core/CircularProgress';
import ListItem from '@material-ui/core/ListItem';
import deepPurple from '@material-ui/core/colors/deepPurple';
import { withStyles } from '@material-ui/core/styles';
import withRoot from '../withRoot';
import {ACTIVE_LEARNING_SERVER, TURK_CONDITION_SHOW_PREDICTIONS} from '../configuration';
import {get_user_url, is_valid, post_data} from '../utils';

const styles = theme => ({
  internal: {
    padding: theme.spacing.unit * 2,
  },
  form: {
    border: "1px solid black",
    padding: theme.spacing.unit * 2,
    marginTop: theme.spacing.unit * 4,
    // textAlign: "center",
  },
  survey_question: {
    // fontWeight: "bold",
  },
  textField: {
    marginRight: theme.spacing.unit * 2,
  }
});

const LIKERT_OPTIONS = [
  "Strongly Disagree",
  "Disagree",
  "Neutral",
  "Agree",
  "Strongly Agree",
];

const PROBABILISIC_OPTIONS = [
  "Almost Never True", //rarely
  "Usually Not True", //ocassionaly
  "Occasionally True", /// consistently
  "Usually True", //
  "Almost Always True",
]

const NUM_TOP_ENTS = 3;

function is_valid_likert_val(val) {
  return is_valid(val) && val >= 0 && val < 5;
}


class LikertScaleClass extends React.Component {
  state = {
    selected_option: null,
  };

  handleChange(event) {
    const val = parseInt(event.target.value);
    this.props.set_val(val);
    this.setState({selected_option: event.target.value});
  }

  render() {
    const {options, likert_type} = this.props;
    const {selected_option} = this.state;

    const radio_buttons = (likert_type === "probabilistic"? options : options).map(
      (option, index) => {
      const label = <div style={{textAlign: "center"}}>
          {option}
      </div>;
      return <FormControlLabel
        key={`radio_group_control_${option}_${index}`}
        value={`${index}`}
        control={<Radio color="primary" />}
        label={label}
        labelPlacement="top"
      />;
    });
    return (
      <div>
        {/* <FormLabel component="legend">labelPlacement</FormLabel> */}
        <RadioGroup
          aria-label="position"
          name="position"
          value={selected_option}
          onChange={this.handleChange.bind(this)}
          row>
          {radio_buttons}
        </RadioGroup>
      </div>
    );
  }
}

LikertScaleClass.propTypes = {
  classes: PropTypes.object.isRequired,

  // 5 length
  // 0 corresponds to text for 1
  // 4 corresponds to text for 4
  options: PropTypes.array,
  set_val: PropTypes.func,

  likert_type: PropTypes.string,
};

const LikertScale = withRoot(withStyles(styles)(LikertScaleClass));

class TopEntitiesClass extends React.Component {
  state = {
    text_fields_value: [null, null, null],
    is_valid_values: [false, false, false],
  };

  componentDidMount() {
    const {top_entities} = this.props;
    const text_fields_value = [];
    const is_valid_values = [];
    for (let i = 0; i < top_entities.length; i++) {
      text_fields_value.push(null);
      is_valid_values.push(false);
    }

    this.setState({text_fields_value, is_valid_values});
    if (is_valid_values.every(x => x)) {
      this.props.set_val(true);
    }
  }

  componentWillReceiveProps(newProps) {
    const {top_entities} = newProps;
    const text_fields_value = [];
    const is_valid_values = [];
    if (newProps.top_entities !== this.props.top_entities) {
      for (let i = 0; i < top_entities.length; i++) {
        text_fields_value.push(null);
        is_valid_values.push(false);
      }

      this.setState({text_fields_value, is_valid_values});
      this.props.set_val(is_valid_values.every(x => x));
    }
  }

  updateTextField = index => event => {
    const {top_entities} = this.props;
    const {text_fields_value, is_valid_values} = this.state;
    text_fields_value[index] = event.target.value;
    is_valid_values[index] |= event.target.value.toLowerCase() === top_entities[index].toLowerCase();
    this.props.set_val(is_valid_values.every(x => x));
    this.setState({text_fields_value, is_valid_values});
  }

  render() {
    const {classes} = this.props;
    const {text_fields_value, is_valid_values} = this.state;
    const text_fields = [];
    for (let i = 0; i < is_valid_values.length; i++) {
      const props = {error: !is_valid_values[i]};
      text_fields.push(
          <TextField
            label={`Top Ent #${i + 1}`}
            className={classes.textField}
            // value={text_field_value}
            onChange={this.updateTextField(i).bind(this)}
            margin="normal"
            variant="outlined"
            {...props}
        />
      );
    }

    return <div>
      {text_fields}
    </div>
  }
}

TopEntitiesClass.propTypes = {
  classes: PropTypes.object.isRequired,
  top_entities: PropTypes.array.isRequired,
  set_val: PropTypes.func,
}

const TopEntities = withRoot(withStyles(styles)(TopEntitiesClass));

/**
 * 
 * Identify top 3 entities (comprehension of side panel)
 * Likert Scale (1 - 5) How confident are you in these entities?
 * Likert Scale (1 - 5) How confident are you in your models predictions (estimate precision)?
 * Likert Scale (1 - 5) Would another round of labeling improve your predictions?
 * Additional comments / feedback
 */

class TurkSurvey extends React.Component {
  state = {
      top_ents_valid: false,
      ent_conf_ls: -1,
      pred_conf_ls: -1,
      progress_further_ls: -1,
      comments: null,
      submitting: false,
      submitted: false,
      survey_code: null,
      error: null,
  };

  submit_survey() {
    const {top_entities} = this.props;
    const {comments, ent_conf_ls, pred_conf_ls, progress_further_ls} = this.state;

    if (this.submit_is_valid()) {
      const submit_data = {comments, ent_conf_ls, pred_conf_ls, progress_further_ls, top_entities, TURK_CONDITION_SHOW_PREDICTIONS};
      const url = get_user_url(
          `${ACTIVE_LEARNING_SERVER}/api/submit_turk_survey/`,
          {},
      );

      post_data(url, {submit_data}).then((response) => {
        const {survey_code} = response;
        if (!is_valid(survey_code)) {
          this.setState({error: "Error generating survey code ocurred, please reach out on Turk"});
        } else {
          this.setState({
            survey_code,
            submitting: false,
            submitted: true,
          });
        }
      });

      this.setState({submitting: true});
    }
  }

  add_comments(event) {
    this.setState({comments: event.target.value});
  }

  submit_is_valid() {
    const {top_ents_valid, ent_conf_ls, pred_conf_ls, progress_further_ls} = this.state;

    return (
      top_ents_valid &&
      is_valid_likert_val(ent_conf_ls) &&
      is_valid_likert_val(pred_conf_ls) &&
      is_valid_likert_val(progress_further_ls)
    );
  }

  set_form_val = val_name => val => {
    const state = {};
    state[val_name] = val;
    this.setState({...state});
  }

  survey_code_message() {
    const {survey_code} = this.state;

    return <div>
        <Typography variant="h6">
        Congratualtions here is your survey code:
        </Typography>
          <Typography color="error" style={{margin: 4, textAlign: "center"}} variant="h6">
            {survey_code}
         </Typography>
         <Typography variant="h6">
         Please submit this code to get credit. Note: this code is only valid for you.
        </Typography>
    </div>;
  }

  render() {
    const {classes, top_entities} = this.props;
    const {submitted, submitting, error} = this.state;
    const can_submit = this.submit_is_valid();
    const buttonProps = {
      disabled: !can_submit
    };

    if (is_valid(error)) {
      return <div className={classes.internal}>
        <Typography color="error" variant="h4">
          {error}
        </Typography>
      </div>;
    }

    return <div className={classes.internal}>
      {submitted ? this.survey_code_message() : null}
      {submitting ? <CircularProgress /> : null}
      {!submitted && !submitting ?
      <div>
        <Typography variant="h4">
          Task Completion Survey!
        </Typography>
        <Typography variant="body1">
          You have now completed the task! Fill out the survey below to recieve your
          survey code.
        </Typography>
        <div className={classes.form}>
          <Typography variant="subtitle2">
            To verify your form, identify the top {top_entities.length} entities from the Predicted Entities list at the bottom of the "Model Results" side panel.
          </Typography>
          <TopEntities top_entities={top_entities} set_val={this.set_form_val('top_ents_valid').bind(this)} />
          <Typography className={classes.survey_question} variant="subtitle1">
            I think the top predicted {top_entities.length} entities are correctly identified.
          </Typography>
          <br />
          <LikertScale
            set_val={this.set_form_val('ent_conf_ls').bind(this)}
            options={[
              "Strongly Disagree",
              "Disagree",
              "Neutral",
              "Agree",
              "Strongly Agree",
            ]}
          />
          <br />
          <Typography className={classes.survey_question} variant="subtitle1">
            The computer program/model is correctly able to identify the adverse drug reactions in the drug reviews.
          </Typography>
          <br />
          <LikertScale
            set_val={this.set_form_val('pred_conf_ls').bind(this)}
            options={[
              "Almost Never True", //rarely
              "Usually Not True", //ocassionaly
              "Occasionally True", /// consistently
              "Usually True", //
              "Almost Always True",
            ]}
            likert_type='probabilistic'
          />
          <br />
          <Typography className={classes.survey_question} variant="subtitle1">
            On the panel to right to the bottom, there is a predicted entities list. How would another round of labeling would improve the contents or the ranking of the top 10 entities in the predicted entities list.
          </Typography>
          <br />
          <LikertScale
            set_val={this.set_form_val('progress_further_ls').bind(this)}
            options={[
              "All of them would change",
              "",
              "Half of them would change",
              "",
              "None of them or one of them would change",
            ]}
          />
          <br />
          <TextField
            id="outlined-multiline-flexible"
            label="Comments/Feedback"
            multiline
            fullWidth
            margin="normal"
            helperText="Fill in any additional comments or feedback you have from your experience with this tool"
            variant="outlined"
            onChange={this.add_comments.bind(this)}
        />
          <Button
            variant="contained"
            color="primary"
            onClick={this.submit_survey.bind(this)}
            {...buttonProps}
          >
            Submit Survey!
          </Button>
        </div>
      </div>
      : null}
    </div>;
  }
}

TurkSurvey.propTypes = {
  classes: PropTypes.object.isRequired,
  buttonProps: PropTypes.object,
  top_entities: PropTypes.array,
};

export default withRoot(withStyles(styles)(TurkSurvey));
