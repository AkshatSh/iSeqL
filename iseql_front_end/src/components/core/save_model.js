import React from 'react';
import PropTypes from 'prop-types';
import Button from '@material-ui/core/Button';
import CircularProgress from '@material-ui/core/CircularProgress';
import ListItem from '@material-ui/core/ListItem';
import deepPurple from '@material-ui/core/colors/deepPurple';
import { withStyles } from '@material-ui/core/styles';
import withRoot from '../../withRoot';
import {ACTIVE_LEARNING_SERVER} from '../../configuration';
import {get_user_url} from '../../utils';

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

class SaveModelButton extends React.Component {
  state = {
      is_running: false,
  };

  handleSaveClick() {
    this.setState({is_running: true});
    fetch(
        get_user_url(`${ACTIVE_LEARNING_SERVER}/api/save_model/`)
    ).then(results => (
        results.json()
    )).then(results => {
        this.setState({is_running: false});
    });
  }

  render() {
    const { classes, buttonProps} = this.props;
    const {is_running} = this.state;
    return is_running ? 
        <CircularProgress color="secondary" />
        : 
        <Button
                    color="inherit"
                    onClick={this.handleSaveClick.bind(this)}
                    className={classes.button}
                    {...buttonProps}
            >
            Save Current Model
        </Button>;
        
  }
}

SaveModelButton.propTypes = {
  classes: PropTypes.object.isRequired,
  buttonProps: PropTypes.object,
};

export default withRoot(withStyles(styles)(SaveModelButton));
