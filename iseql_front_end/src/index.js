import React from 'react';
import ReactDOM from 'react-dom';
import Index from './pages/index';
import TurkIndex from './pages/turk_main_page';
import AppBar from '@material-ui/core/AppBar';
import Tabs from '@material-ui/core/Tabs';
import Tab from '@material-ui/core/Tab';
import * as serviceWorker from './serviceWorker';
import Dataset from './pages/dataset';
import DatasetGroundTruth from './pages/dataset_ground_truth_eval';
import { BrowserRouter as Router, Route, Link } from "react-router-dom";

const DatasetClassifer = ({ match }) => (
    <Dataset classifier_class={match.params.class} dataset_id={parseInt(match.params.id)}/>
);

const GroundTruth = ({match}) => (
  <DatasetGroundTruth classifier_class={match.params.class} dataset_id={parseInt(match.params.id)} />
);

const TurkPage = ({match}) => (
  <TurkIndex turk_id={match.params.turk_id}/>
);

const Instruct_Page = ({match}) => (
  <TurkIndex turk_id={null}/>
); 

ReactDOM.render(
<Router>
    <div>
      <Route path="/" exact component={Index} />
      <Route path="/turk/" exact component={Instruct_Page} />
      <Route path="/turk/:turk_id/" exact component={TurkPage} />
      <Route path="/dataset/:id/:class/" component={DatasetClassifer} />
      <Route path="/ground_truth/:id/:class/" component={GroundTruth} />
      <Route path="/users/" component={Index} />
    </div>
  </Router>, document.getElementById('root'));

// If you want your app to work offline and load faster, you can change
// unregister() to register() below. Note this comes with some pitfalls.
// Learn more about service workers: http://bit.ly/CRA-PWA
serviceWorker.unregister();
