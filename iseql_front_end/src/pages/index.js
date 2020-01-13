import React from 'react';
import PropTypes from 'prop-types';
import Button from '@material-ui/core/Button';
import Dialog from '@material-ui/core/Dialog';
import DialogTitle from '@material-ui/core/DialogTitle';
import DialogContent from '@material-ui/core/DialogContent';
import DialogActions from '@material-ui/core/DialogActions';
import Typography from '@material-ui/core/Typography';
import { withStyles } from '@material-ui/core/styles';
import Fab from '@material-ui/core/Fab';
import AddIcon from '@material-ui/icons/Add';
import Collapse from '@material-ui/core/Collapse';
import ExpandLess from '@material-ui/icons/ExpandLess';
import ExpandMore from '@material-ui/icons/ExpandMore';
import List from '@material-ui/core/List';
import ListItem from '@material-ui/core/ListItem';
import ListItemText from '@material-ui/core/ListItemText';
import Divider from '@material-ui/core/Divider';
import withRoot from '../withRoot';
import configuration from '../configuration';
import AppBar from '@material-ui/core/AppBar';
import Toolbar from '@material-ui/core/Toolbar';
import IconButton from '@material-ui/core/IconButton';
import MenuIcon from '@material-ui/icons/Menu';
import LoginButton from '../components/core/login_button';

function ListItemLink(props) {
  return <ListItem button component="a" {...props} />;
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
  grow: {
    flexGrow: 1,
  }
});

class Index extends React.Component {
  state = {
    open: false,
    login_open: false,
    user_name: null,
    list_open: {},
  };

  componentDidMount() {
    this.get_all_session_data()
  }

  handleClose = () => {
    this.setState({
      open: false,
    });
  };

  handleClick = () => {
    this.setState({
      open: true,
    });
  };

  get_all_session_data = () => {
    fetch(configuration.ACTIVE_LEARNING_SERVER + "/api/sessions/")
    .then(results => {
        return results.json();
    }).then(data => { 
        this.setState({session_data : data});
    })
  }

  handle_list_expansion = (dataset_key) => {
    const list_open = this.state.list_open[dataset_key];
    const list_open_data = this.state.list_open;
    list_open_data[dataset_key] = !list_open;
    this.setState({
      list_open: list_open_data,
    });
  };

  create_list() {
      const list_items = []
      const { session_data } = this.state;
      for(var key in session_data) {
        const data = session_data[key]
        const name = data['name'];
        const classes = data['classes']

        const sub_list_items = []
        for (const class_i in classes) {
          const class_name = classes[class_i];
          sub_list_items.push(
            <div>
              <ListItemLink href={`/dataset/${key}/${class_name}/`} className={classes.nested}>
                  <ListItemText inset primary={class_name} />
              </ListItemLink>
              <Divider light />
            </div>
          )
        }

        const dataset_key = "open_dataset_" + key;
        list_items.push(
          <div>
          {/* <ListItemLink href={"/dataset/" + key}>
            <ListItemText primary={name} />
          </ListItemLink> */}
          <ListItem button onClick={() => {this.handle_list_expansion(dataset_key)}}>
            <ListItemText primary={name} />

            {this.state.list_open[dataset_key] ? <ExpandLess /> : <ExpandMore />}
          </ListItem>
          <Divider light />
          <Collapse in={this.state.list_open[dataset_key]} timeout="auto" unmountOnExit>
            <List component="div" disablePadding>
              {sub_list_items}
            </List>
          </Collapse>
          </div>
        );
      }


      return (
        <List component="nav">
          {list_items}
        </List>
      );
  }

  handleLogin(user_name) {
    this.setState({user_name});
  }

  render() {
    const { classes } = this.props;
    const { open, login_open, user_name } = this.state;

    const disabled = user_name === null;
    return (
      <div>
          <div className={classes.grow}>
            <AppBar position="static">
            <Toolbar>
              <span className={classes.grow} />
              <LoginButton user_name={user_name} handleSetUserName={this.handleLogin.bind(this)}/>
            </Toolbar>
          </AppBar>
        </div>
        <div className={classes.root}>
          <Dialog open={open} onClose={this.handleClose}>
            <DialogTitle>Select a dataset</DialogTitle>
            <DialogContent>
              {this.create_list()}
              <Fab color="primary" aria-label="Add" size="small" className={classes.fab}>
                <AddIcon />
              </Fab>
            </DialogContent>
            <DialogActions>
              <Button color="primary" onClick={this.handleClose}>
                OK
              </Button>
            </DialogActions>
          </Dialog>
          <Typography variant="h4" gutterBottom>
            iSeqL: Interactive Sequence Learning
          </Typography>
          <Typography variant="subtitle1" gutterBottom>
            A tool to help build named entity and sequence labeling classifiers
          </Typography>
          <Button variant="contained" color="primary" onClick={this.handleClick} disabled={disabled}>
            Datasets
          </Button>
        </div>
      </div>
    );
  }
}

Index.propTypes = {
  classes: PropTypes.object.isRequired,
};

export default withRoot(withStyles(styles)(Index));
