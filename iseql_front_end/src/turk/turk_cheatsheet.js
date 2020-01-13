import React from 'react';
import PropTypes from 'prop-types';
import Typography from '@material-ui/core/Typography';
import Button from '@material-ui/core/Button';
import Radio from '@material-ui/core/Radio';
import Paper from '@material-ui/core/Paper';
import Dialog from '@material-ui/core/Dialog';
import DialogTitle from '@material-ui/core/DialogTitle';
import DialogActions from '@material-ui/core/DialogActions';
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
import {ACTIVE_LEARNING_SERVER} from '../configuration';
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
  },
  button: {
      margin: 10,
  },
  cheatsheet: {
      padding: theme.spacing.unit * 4,
      margin: 10,
      minWidth: theme.spacing.unit * 100,
  },
});


class TurkCheatsheet extends React.Component {
  state = {
    cheatsheet_open: false,
  };

  create_cheatsheet() {
      const {classes} = this.props;
      return (
        <Paper className={classes.cheatsheet}>
            <div>
              <Typography align='center' variant="h6" gutterBottom>
                Definitions
              </Typography>
              <Typography variant='body1' gutterBottom>
                <b>Adverse Drug Reaction:</b> Any Symptom that happens during or after taking the drug, that is unintentional. <em>For example: I took Lipitor, but started feeling very <u>nauseous</u>. Nauseous is an adverse reaction in this example.</em>
              </Typography>
            </div>
            <div>
              <Typography align='center' variant="h6" gutterBottom>
                Interactions
              </Typography>
              <Typography variant='body1' gutterBottom>
                <b>Highlight text to label</b>
              </Typography>
              <Typography variant='body1' gutterBottom>
                <b>Click on highlighted text to unlabel</b>
              </Typography>
            </div>
            <div>
            <Typography align='center'  variant="h6" gutterBottom>
                Labeling Guidelines
              </Typography>
              <Typography variant='body1' gutterBottom>
                <b>Be careful of negations:</b> An adverse reaction is only one that the user experienced. <em>For example: I took Lipitor, but felt no pain nor any nausea. Has no adverse reactions, since the user did not experience any.</em>
              </Typography>
              <Typography variant='body1' gutterBottom>
                <b>Reactions are not only physical:</b> Reactions occur to many people in different forms, we are trying to identify all the different reactions, people have <em>For example: I took Lipitor, but became very <u>depressed</u> and had <u>suicidal thoughts</u>. The reactions here are depressed and suicidal thoughts, since they appear to be adverse reactions even though they are not physical reactions.</em>
              </Typography>
              <Typography variant='body1' gutterBottom>
                <b>Single Body Part Attachment is Fine:</b>When labeling reactions, if the reaction is specific to a single body part, it is a good idea to include the body part. <em>For example: I took Lipitor, but started experience severe <u>abdominal pain</u>. The pain here unambigiously refers to a single body part, so it is fine to include.</em>
              </Typography>
              <Typography variant='body1' gutterBottom>
                <b>Ignore  Unnecessary Adjectives:</b> When labeling adverse reactions, it is a good idea to avoid uncessary adjectives. The aim of this task is to get the core reactions in simple entities. <em>For example: I took Lipitor, but started experience severe <u>abdominal pain</u>. Abdominal Pain is an adverse reaction in this example, however severe is not necessary. Another example: I am experiencing very <u>slow walking</u>. Here slow is not an adverse reaction, walking is not an advere reaction, but slow walking is, however the term very is not necessary so it is excluded.</em>
              </Typography>
              <Typography variant='body1' gutterBottom>
                <b>Be concise:</b> When possible it is best to get the reaction and the body part it is describing, however in some cases this may be incredibly long. It is best to only get the reaction in this case. <em>For example: I took Lipitor, but started feeling very <u>pain</u> in my arms , legs , and stomach. The adverse reaction in this case is just pain, because there is no single body part to tie this too</em>
              </Typography>
            </div>
          </Paper>
      );
  }

  handleOpen() {
      this.setState({cheatsheet_open: true});
      fetch(
          get_user_url(`${ACTIVE_LEARNING_SERVER}/api/used_cheatsheet/`)
      ).then(results => (
          results.json()
      )).then(_ => {
          return;
      });
  }

  handleClose() {
      this.setState({cheatsheet_open: false});
  }

  render() {
    const {classes} = this.props;
    const {cheatsheet_open} = this.state;
    return (
      <div>
        <Button
            color="inherit"
            onClick={this.handleOpen.bind(this)}
            className={classes.button}
            >
                CHEAT SHEET
            </Button>
            <Dialog maxWidth='lg' open={cheatsheet_open} onClose={this.handleClose.bind(this)}>
              <DialogTitle style={{textAlign: 'center'}}>Cheat Sheet</DialogTitle>
              {this.create_cheatsheet()}
            </Dialog>
      </div>
    );
  }
}

TurkCheatsheet.propTypes = {
  classes: PropTypes.object.isRequired,
};

export default withRoot(withStyles(styles)(TurkCheatsheet));