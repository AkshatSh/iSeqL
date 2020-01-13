import React from 'react';
import PropTypes from 'prop-types';
import Table from '@material-ui/core/Table';
import TableBody from '@material-ui/core/TableBody';
import TableCell from '@material-ui/core/TableCell';
import TableHead from '@material-ui/core/TableHead';
import TablePagination from '@material-ui/core/TablePagination';
import TableRow from '@material-ui/core/TableRow';
import TableSortLabel from '@material-ui/core/TableSortLabel';
import Paper from '@material-ui/core/Paper';
import Tooltip from '@material-ui/core/Tooltip';
import { withStyles } from '@material-ui/core/styles';
import withRoot from '../../withRoot';
import {is_valid} from '../../utils';

function _desc(a, b, orderBy) {
    if (!is_valid(b[orderBy])) {
      if (!is_valid(a[orderBy])) {
        return 0;
      }

      return 1;
    }

    if (!is_valid(a[orderBy]) && is_valid(b[orderBy])) {
      return -1;
    }

    if (b[orderBy] < a[orderBy]) {
        return -1;
    }
    if (b[orderBy] > a[orderBy]) {
        return 1;
    }
    return 0;
}

function desc(a, b, orderBy) {
  const res = _desc(a, b, orderBy);
  if (orderBy.indexOf("rank") >= 0) {
    return -res;
  }

  return res;
}

function stableSort(array, cmp) {
    const stabilizedThis = array.map((el, index) => [el, index]);
    stabilizedThis.sort((a, b) => {
        const order = cmp(a[0], b[0]);
        if (order !== 0) return order;
        return a[1] - b[1];
    });
    return stabilizedThis.map(el => el[0]);
}

function getSorting(order, orderBy) {
    return order === 'desc' ? (a, b) => desc(a, b, orderBy) : (a, b) => -desc(a, b, orderBy);
}

const styles = theme => ({
  root: {
    width: '100%',
    marginTop: theme.spacing.unit * 3,
  },
  table: {
    minWidth: 1020,
  },
  tableWrapper: {
    overflowX: 'auto',
  },
});

class SortTableHead extends React.Component {
    render() {
        const { onSelectAllClick, order, orderBy, numSelected, rowCount, onRequestSort, rows } = this.props;
        const createSortHandler = property => event => {
          onRequestSort(event, property);
        };
      
        return (
          <TableHead>
            <TableRow>
              {/* <TableCell padding="checkbox">
                <Checkbox
                  indeterminate={numSelected > 0 && numSelected < rowCount}
                  checked={numSelected === rowCount}
                  onChange={onSelectAllClick}
                />
              </TableCell> */}
              {rows.map(
                row => (
                  <TableCell
                    key={row.id}
                    numeric={row.numeric}
                    padding={row.disablePadding ? 'none' : 'default'}
                    sortDirection={orderBy === row.id ? order : false}
                  >
                    <Tooltip
                      title="Sort"
                      placement={row.numeric ? 'bottom-end' : 'bottom-start'}
                      enterDelay={300}
                    >
                      <TableSortLabel
                        active={orderBy === row.id}
                        direction={order}
                        onClick={createSortHandler(row.id)}
                      >
                        {row.label}
                      </TableSortLabel>
                    </Tooltip>
                  </TableCell>
                ),
                this,
              )}
            </TableRow>
          </TableHead>
        );
      }
}
/**
 * Props:
 *  rows:
 *  const rows = [
  { id: 'name', numeric: false, disablePadding: true, label: 'Dessert (100g serving)' },
  { id: 'calories', numeric: true, disablePadding: false, label: 'Calories' },
  { id: 'fat', numeric: true, disablePadding: false, label: 'Fat (g)' },
  { id: 'carbs', numeric: true, disablePadding: false, label: 'Carbs (g)' },
  { id: 'protein', numeric: true, disablePadding: false, label: 'Protein (g)' },
]
 */

SortTableHead.propTypes = {
    onSelectAllClick: PropTypes.object,
    order: PropTypes.object,
    orderBy: PropTypes.object,
    numSelected: PropTypes.object,
    rowCount: PropTypes.object,
    onRequestSort: PropTypes.object,
    rows: PropTypes.array,
}

class SortableTable extends React.Component {
  state = {
      order: 'asc',
      orderBy: 'id',
      selected: [],
      data: [],
      page: 0,
      rowsPerPage: 10,
  };

  componentDidMount() {
    const {defaultOrder, defaultOrderBy} = this.props;
    if (is_valid(defaultOrder)) {
      this.setState({order: defaultOrder});
    }

    if (is_valid(defaultOrderBy)) {
      this.setState({orderBy: defaultOrderBy});
    }
  }

  handleRequestSort(event, property) {
    const {orderBy, order} = this.state;
    const isDesc = orderBy === property && order === 'desc';
    this.setState({
      order: isDesc ? 'asc' : 'desc',
      orderBy: property,
    });
  }

  handleSelectAllClick(event) {
    const {data} = this.props;
    if (event.target.checked) {
      const newSelecteds = data.map(n => n.id);
      this.setState({selected: newSelecteds});
      return;
    }
    this.setState({selected: []});
  }

  handleClick = id => ((event, id) => {
    const {selected} = this.state;
    const selectedIndex = selected.indexOf(id);
    let newSelected = [];

    if (selectedIndex === -1) {
      newSelected = newSelected.concat(selected, id);
    } else if (selectedIndex === 0) {
      newSelected = newSelected.concat(selected.slice(1));
    } else if (selectedIndex === selected.length - 1) {
      newSelected = newSelected.concat(selected.slice(0, -1));
    } else if (selectedIndex > 0) {
      newSelected = newSelected.concat(
        selected.slice(0, selectedIndex),
        selected.slice(selectedIndex + 1),
      );
    }

    this.setState({selected: newSelected});
  });

  handleChangePage(event, page) {
    this.setState({page});
  }

  handleChangeRowsPerPage(event) {
    this.setState({rowsPerPage: event.target.value});
  }

  render() {
    const {row_header, classes, data} = this.props;
    const {order, orderBy, selected, page, rowsPerPage} = this.state;
    const isSelected = id => selected.indexOf(id) !== -1;
    const emptyRows = rowsPerPage - Math.min(rowsPerPage, data.length - page * rowsPerPage);
    return <Paper className={classes.root}>
            <div className={classes.tableWrapper}>
            <Table className={classes.table} aria-labelledby="tableTitle">
                <SortTableHead
                    numSelected={selected.length}
                    order={order}
                    orderBy={orderBy}
                    onSelectAllClick={this.handleSelectAllClick.bind(this)}
                    onRequestSort={this.handleRequestSort.bind(this)}
                    rowCount={data.length}
                    rows={row_header}
                />
                <TableBody>
                {stableSort(data, getSorting(order, orderBy))
                    .slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage)
                    .map((n, index) => {
                    const isItemSelected = isSelected(n.id);
                    const cells = [];
                    for (let i = 0; i < row_header.length; i++) {
                        const rh = row_header[i];
                        let rprops = {key: `${index}_table_cell_${rh.id}_${n[rh.id]}_${n}`, numeric: rh.numeric}
                        if (n[rh.id] === 'id') {
                          rprops.component = "th";
                          rprops.scope = "row";
                          rprops.padding = "none";
                        }
                        cells.push(
                            <TableCell {...rprops}>
                                {n[rh.id]}
                            </TableCell>
                        );
                    }
                    return (
                        <TableRow
                        hover
                        onClick={event => this.handleClick.bind(this)(event, n.id)}
                        role="checkbox"
                        aria-checked={isItemSelected}
                        tabIndex={-1}
                        key={n.id}
                        selected={isItemSelected}
                        >
                        {cells}
                        </TableRow>
                    );
                    })}
                {emptyRows > 0 && (
                    <TableRow style={{ height: 49 * emptyRows }}>
                    <TableCell colSpan={6} />
                    </TableRow>
                )}
                </TableBody>
            </Table>
            </div>
            <TablePagination
                rowsPerPageOptions={[5, 10, 25]}
                component="div"
                count={data.length}
                rowsPerPage={rowsPerPage}
                page={page}
                backIconButtonProps={{
                    'aria-label': 'Previous Page',
                }}
                nextIconButtonProps={{
                    'aria-label': 'Next Page',
                }}
                onChangePage={this.handleChangePage.bind(this)}
                onChangeRowsPerPage={this.handleChangeRowsPerPage.bind(this)}
            />
        </Paper>;   
  }
}



SortableTable.propTypes = {
  classes: PropTypes.object.isRequired,
  row_header: PropTypes.array,
  data: PropTypes.array,
  defaultOrder: PropTypes.string,
  defaultOrderBy: PropTypes.string,
};

export default withRoot(withStyles(styles)(SortableTable));
