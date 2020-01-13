import React from 'react';
import PropTypes from 'prop-types';
import Button from '@material-ui/core/Button';
import Dialog from '@material-ui/core/Dialog';
import TextField from '@material-ui/core/TextField';
import DialogTitle from '@material-ui/core/DialogTitle';
import DialogActions from '@material-ui/core/DialogActions';
import Avatar from '@material-ui/core/Avatar';
import deepOrange from '@material-ui/core/colors/deepOrange';
import List from '@material-ui/core/List';
import ListItem from '@material-ui/core/ListItem';
import ListItemText from '@material-ui/core/ListItemText';
import Divider from '@material-ui/core/Divider';
import deepPurple from '@material-ui/core/colors/deepPurple';
import { withStyles } from '@material-ui/core/styles';
import Cookies from 'universal-cookie';
import withRoot from '../../withRoot';
import {COOKIE_LOGIN, ACTIVE_LEARNING_SERVER} from '../../configuration';
import {post_data} from '../../utils';

function ListItemLink(props) {
  return <ListItem button component="a" {...props} />;
}

const styles = theme => ({
  avatar: {
    margin: 10,
  },
  nested: {
    paddingLeft: theme.spacing.unit * 4,
  },
  fab: {
    margin: theme.spacing.unit,
  },
  grow: {
    flexGrow: 1,
  },
  textField: {
    marginLeft: theme.spacing.unit,
    marginRight: theme.spacing.unit,
    width: 200,
  },
  purpleAvatar: {
    margin: 10,
    color: '#fff',
    backgroundColor: deepPurple[400],
  },
  button: {
    margin: 10,
  },
  selected_user: {
    backgroundColor: deepPurple[100],
  }
});

class LoginButton extends React.Component {
  state = {
    login_open: false,
    user_name: null,
    cookies: new Cookies(),
    selected_user: -1,
    all_users: [],
  };

  handleLoginClick = () => {
    this.setState({
      login_open: true,
    });
  }

  handleLoginClose = () => {
    const {user_name, cookies} = this.state;
    const {handleSetUserName} = this.props;
    if (user_name === null) {
      return;
    }
  
    this.setState({
      login_open: false,
    });

    cookies.set(COOKIE_LOGIN, user_name, {path : '/'});
    const url = `${ACTIVE_LEARNING_SERVER}/api/add_users/`
    post_data(url, {user_name});
    this.setState({user_name});
    handleSetUserName(user_name);
  };

  handleChange = name => event => {
    if (event.target.value.length > 0){
      this.setState({ ...this.state, [name]: event.target.value , selected_user: -1});
    }
  };
  
  componentDidMount() {
    this.mount = true;
    fetch(
      `${ACTIVE_LEARNING_SERVER}/api/users/`,
    ).then(results => {
      return results.json();
    }).then(results => { 
      const all_users = results;
      if (this.mount) {
        this.setState({all_users});
      }
    });
  }

  componentWillUnmount() {
    this.mount = false;
  }

  loginUser = (user_name, selected_user) => () => {
    this.setState({user_name, selected_user});
  }

  getUserList() {
    const {all_users, selected_user} = this.state;
    const {classes} = this.props;
    const list_items = [];
    for (const ui in all_users) {
      const user_name = all_users[ui];
      const class_name = ui === selected_user ? classes.selected_user : null;
      list_items.push(
        <div>
          <ListItemLink onClick={this.loginUser(user_name, ui).bind(this)} className={class_name}>
              <ListItemText inset primary={user_name} />
          </ListItemLink>
          <Divider light />
        </div>
      );
    }

    return list_items;
  }

  render() {
    const { classes, user_name, loginProps} = this.props;
    const { login_open } = this.state;
    return user_name === null || user_name === undefined || user_name.length === 0 ? (
        <div>
            <Button
                    color="inherit"
                    onClick={this.handleLoginClick}
                    className={classes.button}
                    {...loginProps}
            >
                Login
            </Button>
            <Dialog open={login_open} onClose={this.handleLoginClose}>
              <DialogTitle style={{textAlign: 'center'}}>Login</DialogTitle>
              <Divider light />
              {/* <List component="div" disablePadding>
                {this.getUserList()}
              </List> */}
              <DialogActions>
                <TextField
                  id="user-name"
                  label="User Name"
                  className={classes.textField}
                  variant="outlined"
                  onChange={this.handleChange('user_name').bind(this)}
                  margin="normal"
                  style={{display: 'block'}}
                />
                <Button color="primary" onClick={this.handleLoginClose}>
                    Sign In
                </Button>
              </DialogActions>
            </Dialog>
        </div>
    ) : <Avatar className={classes.purpleAvatar}>{user_name[0]}</Avatar>;
  }
}

LoginButton.propTypes = {
  classes: PropTypes.object.isRequired,
  handleSetUserName: PropTypes.func,
  user_name: PropTypes.string,
  loginProps: PropTypes.object,
};

export default withRoot(withStyles(styles)(LoginButton));
