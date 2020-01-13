import React from 'react';
import PropTypes from 'prop-types';
import Button from '@material-ui/core/Button';
import RadioGroup from '@material-ui/core/RadioGroup';
import Radio from '@material-ui/core/Radio';
import FormControlLabel from '@material-ui/core/FormControlLabel';
import Typography from '@material-ui/core/Typography';
import { withStyles } from '@material-ui/core/styles';
import Checkbox from '@material-ui/core/Checkbox';
import Paper from '@material-ui/core/Paper';
import Divider from '@material-ui/core/Divider';
import Tooltip from '@material-ui/core/Tooltip';
import Help from '@material-ui/icons/Help';
import AppBar from '@material-ui/core/AppBar';
import Toolbar from '@material-ui/core/Toolbar';
import withRoot from '../withRoot';
import configuration from '../configuration';
import LoginButton from '../components/core/login_button';
import {COOKIES, get_user_url, post_data, is_valid, safe_fetch} from '../utils';
import ExampleCard from '../components/exampleCard';
import {COOKIE_LOGIN, ACTIVE_LEARNING_SERVER, TURK_DATASET, TURK_CLASS, TURK_MAIN_PAGE_DATA, TURK_CONDITION_SHOW_PREDICTIONS} from '../configuration';


// images
import UserDiagramImage from '../figs/transparent_user_diagram.png';
import SidePanelImage from '../figs/side_panel.png';
import MainPageImage from '../figs/wrong_main_page_with_predictions.png';
import LabelingGif from '../figs/labeling_demo.gif';
import SubmitGif from '../figs/submit_demo.gif';

import UpdateModelPredGif from '../figs/update_with_predictions_new.gif';
import UpdateModelNoPredGif from '../figs/update_without_predictions.gif';
import { defaultTitleFormatter } from 'vega-lite/build/src/fielddef';

const UpdateModelGif = TURK_CONDITION_SHOW_PREDICTIONS ? UpdateModelPredGif : UpdateModelNoPredGif;


const styles = theme => ({
  root: {
    // textAlign: 'center',
    paddingTop: theme.spacing.unit * 4,
    marginLeft: theme.spacing.unit * 30,
    marginRight: theme.spacing.unit * 30,
  },
  nested: {
    paddingLeft: theme.spacing.unit * 2,
  },
  fab: {
    margin: theme.spacing.unit,
  },
  grow: {
    flexGrow: 1,
  },
  cheatsheet: {
    padding: theme.spacing.unit * 4,
    margin: 10,
  },
  divider: {
    marginTop: theme.spacing.unit * 4,
    marginBottom: theme.spacing.unit * 4,
  }
});

class TurkQuizClass extends React.Component {
  state = {
    quiz_data: [
      {
        question: 'In the sentence: "I took lipitor, and started having severe left leg pain", what are the adverse drug reactions entities?',
        options: [
          "a) lipitor",
          "b) leg pain",
          "c) severe left leg pain",
          "d) left leg pain",
        ],
        answer: 1,
        explanation: "a) is the name of the drug, b) is correct. c) contains an adjective, d) contains an adjective.",
      },
      {
        question: "When my model is confident, the number of flipped labels will...",
        options: [
          "a) Go towards 0.",
          "b) Go towards infinity.",
          "c) Become negative.",
          "d) There is no flipped label section.",
        ],
        answer: 0,
        explanation: "a) As stated, when the data stops flipping, the model is more confident and thus is performing well."
      },
      {
        question: "When do I achieve the best quality score?",
        options: [
          "a) 0.0",
          "b) -5",
          "c) 30",
          "d) 1.0",
        ],
        answer: 3,
        explanation: "The quality score is on a range from 0.0 - 1.0, and performs best when at 1.0 . d) was the correct answer."
      }
    ],
    
    selected_answers: [-1, -1, -1],
    is_correct: [false, false, false],
    show_errors: false,
  }

  handleChange = index => (event) => {
    const {quiz_data, selected_answers, is_correct} = this.state;
    selected_answers[index] = event.target.value;
    is_correct[index] = quiz_data[index].answer === parseInt(event.target.value);
    this.setState({selected_answers, is_correct});
  }

  answered_all() {
    const {selected_answers} = this.state;
    return selected_answers.every(x => x >= 0);
  }

  handleCheck() {
    const {on_pass, on_fail} = this.props;
    const {is_correct} = this.state;
    this.setState({show_errors: true});
    const passed = is_correct.every(x => x);
    if (passed) {
      on_pass();
    } else {
      on_fail();
    }
  }


  render() {
    const {quiz_data, selected_answers, is_correct, show_errors} = this.state;
    const answered_all = this.answered_all();
    const quiz_question = quiz_data.map((quiz_question_data, index) => {
      const {question, options, explanation} = quiz_question_data;
      const error = show_errors && !is_correct[index];

      const choices = options.map((option, c_index) => {
        const label = <Typography variant='body2'>
            {option}
        </Typography>;
        return <FormControlLabel
          key={`radio_group_control_${option}_${c_index}`}
          value={`${c_index}`}
          control={<Radio style={{paddingRight: 1}} color="primary" />}
          label={label}
          labelPlacement='end'
        />;
      });

      return (
        <div>
          <Typography variant="body1" gutterBottom>
            {index + 1}) {question}
          </Typography>
          <RadioGroup
            aria-label="position"
            name="position"
            value={selected_answers[index]}
            onChange={this.handleChange(index).bind(this)}
            style={{paddingLeft: 40}}
            row>
          {choices}
          {error ? 
            <Typography variant="body1" gutterBottom color='error'>
              {explanation}
            </Typography>
            : null}
        </RadioGroup>
        </div>
      );
    });

    return <div>
      {quiz_question}
      <div>
        <Button
          variant="contained"
          color="primary"
          disabled={!answered_all}
          onClick={this.handleCheck.bind(this)}
        >
          Check Quiz Answers
        </Button>
      </div>
    </div>;
  }
}

TurkQuizClass.propTypes = {
  classes: PropTypes.object.isRequired,
  on_pass: PropTypes.func,
  on_fail: PropTypes.func,
}

const TurKQuiz = withRoot(withStyles(styles)(TurkQuizClass));

function tooltip_def(term, definition) {
  return <span>
    <b>{term}</b>
    <Tooltip
        title={
            <React.Fragment>
                {definition}
            </React.Fragment>
        }
        animation="zoom"
      >
      <Help style={{fontSize: 12, paddingLeft: 2}} />
  </Tooltip>
  </span>;
}



function def_adr(term='adverse drug reaction') {
  return tooltip_def(
    term,
    'Any Symptom that happens during or after taking the drug, that is unintentional. For example: I took Lipitor, but started feeling very nauseous. Nauseous is an adverse reaction in this example.',
  );
}

function def_label(term='label') {
  return tooltip_def(
    term,
    "an annotation associated with each word."
  );
}

function def_postive_label(term='postive label') {
  return tooltip_def(
    term,
    "the word is or belongs to an adverse reaction."
  );
}

function def_negative_label(term='negative label') {
  return tooltip_def(
    term,
    "the word does not belong to an adverse reaction."
  );
}

function def_entity(term='entity') {
  return tooltip_def(
    term,
    "a single adverse reaction. For example: abdominal pain is an entity."
  );
}

function def_model(term='model') {
  return tooltip_def(
    term,
    "a computer program capable of learning patterns from data.",
  );
}

class TurkIndex extends React.Component {
  state = {
    error: false,
    show_answers: false,
    valid_user: false,
    user_name: null,
    quiz_passed: false,
    list_open: {},
  };

  handleLogin(user_name=null) {
    if (!is_valid(user_name)) {
        return;
    }
  
    this.setState({
      login_open: false,
    });

    COOKIES.set(COOKIE_LOGIN, user_name, {path : '/'});
    const url = `${ACTIVE_LEARNING_SERVER}/api/add_users/`
    post_data(url, {user_name});
    this.setState({user_name});
    this.checkIsValidUser();
  };

  handleChange = name => event => {
    this.setState({ ...this.state, [name]: event.target.checked });
  };

  error_callback(_) {
      this.setState({error: true});
  }

  success_callback(results) {
      this.setState({valid_user: results.exists});
  }

  checkIsValidUser() {
    const url = get_user_url(`${ACTIVE_LEARNING_SERVER}/api/has_completed_task/`);
    safe_fetch(url, this.error_callback.bind(this), this.success_callback.bind(this));
  }

  componentDidMount() {
      const {turk_id} = this.props;
      if (is_valid(turk_id)) {
        this.handleLogin(turk_id);
      }
  }

  componentWillReceiveProps(newProps) {
      const {turk_id} = newProps;
      if (turk_id !== this.props.turk_id && is_valid(turk_id)) {
        this.login(turk_id);
      }
  }

  doneLabelFunc() {

  }

  onQuizPass() {
    this.setState({quiz_passed: true});
  }

  onQuizFail() {
    this.setState({quiz_passed: false});
  }

  build_example() {
    const {example_id, example, example_predictions} = TURK_MAIN_PAGE_DATA;
    const {show_answers, quiz_passed} = this.state;
    const dataset_id = TURK_DATASET;
    const classifier_class = TURK_CLASS;
    return (
      <ExampleCard
        key={"example_card_" + example_id}
        classifier_class={classifier_class}
        dataset_id={dataset_id}
        example={example}
        doneLabelFunc={this.doneLabelFunc.bind(this)}
        index={example_id}
        example_prediction={{ranges: example_predictions}}
        show_predictions={show_answers}
      />
    );
  }

  render() {
    const { classes, turk_id } = this.props;
    const { valid_user, show_answers, user_name, quiz_passed } = this.state;

    const disabled = valid_user;
    return (
      <div className={classes.root}>
          {/* <div className={classes.grow}>
            <AppBar position="static">
            <Toolbar>
              <span className={classes.grow} style={{textAlign: 'center'}}>
                  <Typography color='inherit' align='center' variant="h5" gutterBottom>
                    ALSeq: {user_name !== null ? user_name.toUpperCase(): null}
                  </Typography>
              </span>
              <LoginButton user_name={user_name} handleSetUserName={this.handleLogin.bind(this)}/>
            </Toolbar>
          </AppBar> 
        </div> */}
        <div>
          <Typography align='center' variant="h4" gutterBottom>
            HIT: Identifying Adverse Drug Reactions
          </Typography>

          {disabled ?
          <Typography align='center' color="error" variant="subtitle1">
            Sorry, You are no longer eligible for this study. You have already completed this study or a variant.
          </Typography> : null}
          <Typography variant="subtitle1" gutterBottom>
            Welcome to the our study for idenitfying {def_adr()} through our system.
          </Typography>

          <Divider variant='fullWidth' className={classes.divider} light={false}/>
          <Typography align='center' variant='h5' gutterBottom>
            Introduction
          </Typography>
          <Divider variant='fullWidth' className={classes.divider} light={false}/>

          <Typography variant="body1" gutterBottom>
          In the following task, you will label the {def_adr()} in drug reviews to help a computer program create a {def_model()} to predict all the adverse reactions in drug reviews. We anticpate this task should take around <b>60 minutes</b> to complete.
          </Typography>

          {/* <Typography variant="body1" gutterBottom>
          A <b>Machine Learning Model</b> is a computer program capable of learning a pattern to recognize data. You will build this model by providing examples of what are adverse reactions in a series of drug reviews, and the model will learn from what you are labeling, and try to predict what the adverse reactions are on all the examples you have not labeled.
          </Typography> */}

          <Typography variant="body1" gutterBottom>
          <b>Adverse Drug Reaction</b> is Any Symptom that happens during or after taking the drug, that is unintentional. For example: I took Lipitor, but started feeling very nauseous. Nauseous is an adverse reaction in this example.
          </Typography>

          <Typography variant="body1" gutterBottom>
          A <b>Model</b> is a computer program capable of learning patterns from data.
          </Typography>

          <Typography variant='body1' gutterBottom>
          To do this, you will be presented with a series of drug reviews, your task is to identify the adverse reactions in them.
          <ol>
            <li>
            <b>Labeling stage</b>: label text spans that represent {def_adr()}, which will be examples you use to teach the machine. You will go through 5 iterations of labeling 24 examples. After each iteration, the program will learn from your labeled examples, and be evaluated on 1000 drug reviews, use this new information to guide your labeling.
            </li>
            <li>
              <b>Evaluation stage</b>: Once you have labeled all your examples, the computer program will be run on all 1000 reviews on identify the adverse reactions, and produce a summary about them, you will be asked to assess the quality of this program.
            </li>
          </ol>
          </Typography>
          <Typography variant='body1' gutterBottom>
            A <b>label</b> is an annotation associated with each word. <b>Positive</b> means that the word is or belongs to an adverse reaction. For example: I have adbominal pain but no nausea. Abdominal and pain are positive labels. <b>Negative</b> means that the word does not belong to an adverse reaction, in the above example "I, have, but, no, nausea" are all negative. An <b>entity</b> refers to a single adverse reaction. For example: abdominal pain is an entity.
          </Typography>

          <Divider variant='fullWidth' className={classes.divider} light={false}/>
          <Typography align='center' variant='h5' gutterBottom>
            Tool Overview
          </Typography>
          <Divider variant='fullWidth' className={classes.divider} light={false}/>

          {/* <div style={{textAlign: "center", margin: 10}}>
            <img src={UserDiagramImage} alt="user_diagram" style={{width: 600}}/>
            <img src={MainPageImage} alt="main_page_image" style={{width: 600}} />
          </div> */}

          <Typography variant='body1' gutterBottom>
            <b>Label Batch:</b> When you start the task the main tool will open. It will show a series of 24 examples on the left hand side, and a side panel titled Model Results. When you are in this stage, you will go through each of the examples shown on the left, and label all the adverse drug reactions in them. You will be able to label the examples by highlighting the text on the screen. When you have finished, hit the publish examples button, and the {def_model()} will begin to train, and the next batch for you to label will appear. <b>You will go through 5 iterations of labeling 24 examples, to complete the task.</b>
          </Typography>

          <Typography variant='body1' gutterBottom>
            <b>Labeling Interactions</b>
            <p>The main interactions in this application are around labeling</p>
            <ul>
              <li>Highlighting a span of text signifies an {def_entity()}</li>
              <li>Double clicking a word selects a single word</li>
              <li>Click a span of text to unhighlight it</li>
              <li>If you are very unsure about an example, you can hit the three dots below it, and select "exclude"</li>
            </ul>
          </Typography>

          <div style={{textAlign: "center", margin: 10}}>
            <img src={LabelingGif} alt="label_gif" style={{width: 600}}/>
            <Typography variant='caption'>In the above example, you can see what labeling in the system looks like</Typography>
          </div>

          <Divider variant='fullWidth' className={classes.divider} light={false}/>

          <Typography variant='body1' gutterBottom>
            <b>Train Model:</b> In order for the computer program to learn from all the reactions you have labeled. The {def_model()} goes through a process called training, where it tries to learn from your labels. This however can take some time. A progress bar will appear at the top, and the model will train as you label the next batch.
          </Typography>

          <div style={{textAlign: "center", margin: 10}}>
            <img src={SubmitGif} alt="submit_gif" style={{width: 600}} />
            <Typography variant='caption'>Once you have finished labeling, submit the current batch and move to the next one</Typography>
          </div>
          <Typography variant='body1' gutterBottom>
            At somepoint in the middle of the batch, the model will finish its training phase, and a button will appear at the top of the screen titled "Update Model". Clicking this button will update your view to reflect the latest model you have built and its predictions. Click on this button so you can see how the model is progressing. The more examples you label for your model, the better your model will perform. You can now see the model results panel on the left and use it to evaluate your model.
          </Typography>
          <div style={{textAlign: "center", margin: 10}}>
            <img src={UpdateModelGif} alt="update_model_gif" style={{width: 600}} />
            <Typography variant='caption'>The model will finish training during the next batch, and you can update the application to view the results</Typography>
          </div>

          {TURK_CONDITION_SHOW_PREDICTIONS ? 
            <Typography variant='body1'>
            Once you update your model, your model's predictions will be shown underlined in purple, like the example above. You can use this to <b>assess your model performance/predictions</b> and to help <b>guide your labeling</b>. As you progress further and update your model, these predictions will become more and more accurate.
            </Typography>
            : null}

          <Divider variant='fullWidth' className={classes.divider} light={false}/>

          <div style={{display: 'flex'}}>
            <div style={{marginRight: 40}}>
              <img
                src={SidePanelImage}
                alt="side_panel_image"
                style={{display: 'inline-block', height: 500}}
              />
              <Typography align='center' variant='caption' gutterBottom>
                After some iterations the model results side panel will look like
                the above.
                <br />
                <em>Note: The results shown here may not be correct</em>
              </Typography>
            </div>
            <div style={{flexGrow: 1}}>
              <Typography variant='body1' gutterBottom>
                <b>Evaluate Model:</b> On the left hand side there is a view titled "Model Results". This view presents a few informative graphs and lists to help you undestand the {def_model()} you have built so far.
              </Typography>
              <Typography variant='body1' gutterBottom>
                <b>Flipped Labels:</b> This view shows how many labels have changed. It is a count of labels that went from {def_postive_label()} to {def_negative_label()}, and vice versa. The intention of this view is to see how much your model is changing over time. As your model becomes more decisive this should be approaching 0.
              </Typography>
              <Typography variant='body1' gutterBottom>
                <b>Quality Score:</b> One issue with training your model, is that your model can learn to memorize the labels you feed it, however we want to see how well it performs on data it has not seen. This introduces the quality score. This is a score ranging from 0.0 to 1.0 denoting how well this model is performing. There are two scores: Train and Test. Train is all the examples that your model is seen and is learning from. Test are examples that your model hasn't seen. As you progress through training, these quality scores should become more and more accurate. Here a score of 1 signifies a perfect model capable of predicting every adverse reaction. 0 signifies a model that has not learned anything.
              </Typography>
              <Typography variant='body1' gutterBottom>
                <b>Predicted {def_entity("Entities")} Rank Chart:</b> When your model is done training, the model is run all the data (drug reviews), whether you have labeled them or not. Since this is a lot of information, we summarize it by providing the quick view of Predicted Entities. The list shows the most popular entity (how many times it occurs) at the top, and is sorted so the least frequent is at the bottom. To help understand how this graph changes, there are a series of icons to help. A green up arrow means from the previous time you trained this entity has become more popular, a red down arrow means the opposite (it has become less popular). A star signifies that you never labeled a word like this, but the model is guessing it is an adverse reaction. The new icon shows that in the most recent run it predicted this entity, where in previous runs it did not.
              </Typography>
              <Typography variant='body1' gutterBottom>
                <b>Other Entity Rank Charts:</b> In the predicted entity view, you can select the 3 vertical dots to show two other options as well: labeled entities, shows the same information but for everything you have labeled. And discovered entities, shows the same information but for all entities that were predicted but not labeled.
              </Typography>
            </div>
          </div>

          <Typography variant='body1' gutterBottom>
            Since there is a lot of information to remember for this task, we are proving you with the following cheat sheet throughout the task. Just hit the upper left button for cheat sheet, and the cheat sheet below will appear.
          </Typography>

          {/* Cheat Sheet Section */}
          <Paper className={classes.cheatsheet}>
            <Typography align='center' variant="h5" gutterBottom>
              Cheat Sheet
            </Typography>
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
          <Divider variant='fullWidth' className={classes.divider} light={false}/>

          {/* Example Section */}

          <Typography variant="h6" gutterBottom>
            Try out the example below, to get a sense of the task. Hit show answers when you have finished.
          </Typography>

          {this.build_example()}
          <div>
          <FormControlLabel
            control={
              <Checkbox
                color='primary'
                checked={show_answers}
                onChange={this.handleChange('show_answers').bind(this)}
                value="show_answers" />
            }
            label="Show Answers"
          />
          </div>
          <Typography variant="body1" gutterBottom>
            In the above example you can see that in all the occurences of adverse drug reactions are higlighted. Compare your answers with the underlined ones to get a sense of what adverse drug reactions are. For example, in certain cases "foot was so painful" is selected instead of "left foot was so painful". This is to go with the condition that if very specific reactions are labeled, the model will not generalize properly to other reactions.
          </Typography>
          <Divider variant='fullWidth' className={classes.divider} light={false}/>
          <Typography align='center' color="error" variant="h6" gutterBottom>
            If your labeling strategy appears to be inconsisent, you may not get credit for completing this task.
          </Typography>

          <Typography align='center' color="error" variant="h6" gutterBottom>
            This task is also time dependent. It should take under 1 hour, we ask you do this task all in one sitting.
          </Typography>
          
          <Divider variant='fullWidth' className={classes.divider} light={false}/>
          
          {is_valid(turk_id) ? 
          <div>
          <Typography variant="h6" gutterBottom>
            To ensure you are ready to go forward, take the quiz below.
          </Typography>
          <TurKQuiz on_pass={this.onQuizPass.bind(this)} on_fail={this.onQuizFail.bind(this)} />
          {
            quiz_passed ? 
              <Typography variant="h6" gutterBottom>Quiz Passed</Typography>:
              <Typography variant="h6" gutterBottom color='error'>Quiz Not Passed</Typography>
          }

          <Divider variant='fullWidth' className={classes.divider} light={false}/>
          
          <div 
            style={{textAlign: "center"}}>
            {valid_user ? <Typography align='center' color="error" variant="subtitle1">
              You are no longer eligible for this study. You have already completed this study or a variant.
            </Typography> : null}
            {quiz_passed? null : <Typography align='center' color="error" variant="subtitle1">
              You need to pass the quiz to continue.
            </Typography>}
            <Button
              variant="contained"
              color="primary"
              href={`/dataset/${TURK_DATASET}/${TURK_CLASS}/`}
              disabled={disabled || !quiz_passed}
              style={{marginBottom: 40}}
          >
              Start HIT
            </Button>
          </div>
          </div>
          : 
          <Typography variant="h6" gutterBottom>
            Please accept the hit to move forward
          </Typography>
          }
        </div>
      </div>
    );
  }
}

TurkIndex.propTypes = {
  classes: PropTypes.object.isRequired,
  turk_id: PropTypes.object.isRequired,
};

export default withRoot(withStyles(styles)(TurkIndex));