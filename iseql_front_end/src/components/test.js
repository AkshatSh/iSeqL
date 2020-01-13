/* eslint-disable no-console */

import React from 'react';
import PropTypes from 'prop-types';
import classNames from 'classnames';
import { withStyles } from '@material-ui/core/styles';
import TableCell from '@material-ui/core/TableCell';
import TableSortLabel from '@material-ui/core/TableSortLabel';
import Paper from '@material-ui/core/Paper';
import { AutoSizer, Column, SortDirection, Table } from 'react-virtualized';
import MuiVirtualizedTable from './MuiVirtualizedTable';

class ReactVirtualizedTable extends React.Component {
    state = {
        data: [
            ['Frozen yoghurt', 159, 6.0, 24, 4.0],
            ['Ice cream sandwich', 237, 9.0, 37, 4.3],
            ['Eclair', 262, 16.0, 24, 6.0],
            ['Cupcake', 305, 3.7, 67, 4.3],
            ['Gingerbread', 356, 16.0, 49, 3.9],
        ],
        id: 0,
        rows: [],
    };

    createData(dessert, calories, fat, carbs, protein) {
        let {id} = this.state;
        id += 1;
        this.setState({
            id: id,
        });
        return { id, dessert, calories, fat, carbs, protein };
    }

    componentDidMount() {
        const {data} = this.state;
        const rows = [];

        for (let i = 0; i < 200; i += 1) {
            const randomSelection = data[Math.floor(Math.random() * data.length)];
            rows.push(this.createData(...randomSelection));
        }

        this.setState({
            rows: rows,
        });
    }

    columnInfo() {
        return [
            {
              width: 200,
              flexGrow: 1.0,
              label: 'Dessert',
              dataKey: 'dessert',
            },
            {
              width: 120,
              label: 'Calories (g)',
              dataKey: 'calories',
              numeric: true,
            },
            {
              width: 120,
              label: 'Fat (g)',
              dataKey: 'fat',
              numeric: true,
            },
            {
              width: 120,
              label: 'Carbs (g)',
              dataKey: 'carbs',
              numeric: true,
            },
            {
              width: 120,
              label: 'Protein (g)',
              dataKey: 'protein',
              numeric: true,
            },
        ];
    }

    render() {
        const {rows} = this.state;
        const cols = this.columnInfo();
        return (
            <Paper style={{ height: 400, width: '100%' }}>
              <MuiVirtualizedTable
                rowCount={rows.length}
                rowGetter={({ index }) => rows[index]}
                onRowClick={event => console.log(event)}
                columns={cols}
              />
            </Paper>
          );
    }
}

export default ReactVirtualizedTable;