import React from 'react';
import PropTypes from 'prop-types';
import Typography from '@material-ui/core/Typography';
import LinearProgress from '@material-ui/core/LinearProgress';
import { withStyles } from '@material-ui/core/styles';
import withRoot from '../../withRoot';

const styles = theme => ({
    root: {
        margin: theme.spacing.unit * 2,
        fontFamily: 'Roboto',
        textAlign: 'center',
    },
    grow: {
        flexGrow: 1,
    },
});

class Progress extends React.Component {
  render() {
    const {classes, current, total, progress_message} = this.props;
    return (
        <div className={classes.root}>
            {progress_message}
            <LinearProgress
                color="secondary"
                variant="determinate"
                value={current * 100.0 / total}
                className={classes.grow}
            />
        </div>
    );  
  }
}

Progress.propTypes = {
  classes: PropTypes.object.isRequired,
  current: PropTypes.number,
  total: PropTypes.number,
  progress_message: PropTypes.string,
};

export default withRoot(withStyles(styles)(Progress));
