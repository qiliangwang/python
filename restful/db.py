import MySQLdb
from flask import Flask
import flask_restful as restful

app = Flask(__name__)
api = restful.Api(app)

info_data = {
    'localhost_props': {
        'host': '127.0.0.1',
        'user': 'root',
        'passwd': 'password',
        'port': 3306,
    },
    'remote1_props': {
        'host': '192.168.33.206',
        'user': 'dev',
        'passwd': 'lvyue@123456',
        'port': 3316,
    },
    'remote2_props': {
        'host': '192.168.33.206',
        'user': 'dev',
        'passwd': 'lvyue@123456',
        'port': 3326,
    },
    'remote3_props': {
        'host': '192.168.33.206',
        'user': 'dev',
        'passwd': 'lvyue@123456',
        'port': 3336,
    },
    'remote4_props': {
        'host': '192.168.33.206',
        'user': 'dev',
        'passwd': 'lvyue@123456',
        'port': 3346,
    },
    'remote5_props': {
        'host': '192.168.33.206',
        'user': 'dev',
        'passwd': 'lvyue@123456',
        'port': 3356,
    },
    'remote6_props': {
        'host': '192.168.33.206',
        'user': 'dev',
        'passwd': 'lvyue@123456',
        'port': 3366,
    },
}

json_data = {}


class ColumnData(restful.Resource):
    def get(self):
        return {'author': 'VaderWang',
                'version': 1.0,
                'data': json_data,
                }


api.add_resource(ColumnData, '/')


def get_connection(db_name, port_info):
    conn_props = info_data[port_info]
    conn_props['db'] = db_name
    conn = MySQLdb.connect(**conn_props)
    return conn
    pass


def get_dbs(query_info):
    conn = MySQLdb.connect(**info_data[query_info])
    cursor = conn.cursor()
    cursor.execute('SHOW DATABASES;')
    dbs = [db_name[0] for db_name in cursor.fetchall()]
    for db in dbs:
        json_data[query_info][db] = {}
    cursor.close()
    return dbs


def get_tables(db_name, query_info):
    conn_props = info_data[query_info]
    conn_props['db'] = db_name
    conn = MySQLdb.connect(**conn_props)
    cursor = conn.cursor()
    cursor.execute('SHOW TABLES;')
    tables = [table[0] for table in cursor.fetchall()]
    for table in tables:
        sql_statement = "SHOW FULL COLUMNS FROM " + str(table)
        cursor.execute(sql_statement)
        # json_data[query_info][db_name][table] = [column[0] for column in cursor.fetchall()]
        json_data[query_info][db_name][table] = [{'Column': column[0], 'Field': column[1], 'DataSource': query_info, 'DateBase': db_name, 'TableName': table} for column in cursor.fetchall()]
    cursor.close()


def load_data():
    query_list = ['remote1_props', 'remote2_props', 'remote3_props', 'remote4_props', 'remote5_props', 'remote6_props']
    # query_list = ['localhost_props']

    for query_info in query_list:
        print(query_info)
        json_data[query_info] = {}
        dbs = get_dbs(query_info)
        # dbs = ['sell']
        for db in dbs:
            try:
                get_tables(db, query_info)
            except:
                print("No Authority", db)

        pass


if __name__ == '__main__':
    load_data()
    app.run(host='0.0.0.0', port=5000, debug=True)
    # app.run(debug=True)
