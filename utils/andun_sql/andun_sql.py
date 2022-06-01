import pandas as pd
import pymysql
import datetime
import time


class AndunSql:
    def __init__(self, db_type=None):
        self._TEST_DB_HOST = '192.168.100.79'
        self._TEST_DB_ACCOUNT = 'andunroot'
        self._TEST_DB_PASSWORD = 'andun1819'
        self._TEST_DB_PORT = 3307

        # 旧账号
        # self._DB_HOST = 'rm-2ze3zk2327k92186do.mysql.rds.aliyuncs.com'
        # self._DB_ACCOUNT = 'andun_mingkun'
        # self._DB_PASSWORD = 'mpiw9+fMhRMjdMEN'

        # 新账号
        # self._DB_HOST = 'rm-2ze3zk2327k92186do.mysql.rds.aliyuncs.com'
        # self._DB_ACCOUNT = 'python_suanfa'
        # self._DB_PASSWORD = 'dM^G%5onYk&SpY%d$cv$qTyp7QXdbBF7'

        # 新账号
        self._DB_HOST = 'rm-2ze3zk2327k92186do.mysql.rds.aliyuncs.com'
        self._DB_ACCOUNT = 'py_xiongyaokun'
        self._DB_PASSWORD = '%NBug%g0w!ltFR*7j#6a'


    @staticmethod
    def cal_age_by_birthday(birthday_str):
        return int((datetime.datetime.now() - datetime.datetime.strptime(str(birthday_str), '%Y-%m-%d')).days / 365)

    def ansql_read_mysql(self, sql_str):
        db = pymysql.connect(host=self._DB_HOST, user=self._DB_ACCOUNT, password=self._DB_PASSWORD, charset='utf8')
        df = pd.read_sql(sql_str, db, params=None)
        db.close()
        return df

    def ansql_read_mysql_test_db(self, sql_str):
        db = pymysql.connect(host=self._TEST_DB_HOST, user=self._TEST_DB_ACCOUNT, password=self._TEST_DB_PASSWORD, port=self._TEST_DB_PORT, charset='utf8')
        df = pd.read_sql(sql_str, db, params=None)
        db.close()
        return df



    def ansql_bp_feature(self, wear_user_id, date):

        if len(date) > 1:
            # sql_s = "SELECT WEAR_USER_ID,FROMPPG,DATE FROM andun_watch.d_bp_feature_1 WHERE WEAR_USER_ID = '{}' AND DATE in {}".format(wear_user_id, date)
            sql_s_t = "SELECT WEAR_USER_ID,FROMPPG,DATE FROM andun_watch.d_bp_feature WHERE WEAR_USER_ID = '{}' AND DATE in {}".format(wear_user_id, date)
        elif len(date) == 1:
            
            # sql_s = "SELECT WEAR_USER_ID,FROMPPG,DATE FROM andun_watch.d_bp_feature_1 WHERE WEAR_USER_ID = '{}' AND DATE = '{}'".format(wear_user_id, date[0])
            sql_s_t = "SELECT WEAR_USER_ID,FROMPPG,DATE FROM andun_watch.d_bp_feature WHERE WEAR_USER_ID = '{}' AND DATE = '{}'".format(wear_user_id, date[0])
        else:
            return None

        # res_1 = self.ansql_read_mysql(sql_s)
        res_2 = self.ansql_read_mysql(sql_s_t)

        # if len(res_1) >0:
        #     return res_1
        # elif len(res_2) >0:
        #     return res_2
        # else:
        #     return res_1

        return res_2


    def ansql_bp_feature_train(self, wear_user_id, date):

        if len(date) > 1:
            sql_s = "SELECT WEAR_USER_ID,FROMPPG,DATE FROM andun_watch.d_bp_feature_model WHERE WEAR_USER_ID = '{}' AND DATE in {}".format(wear_user_id, date)
        elif len(date) == 1:
            sql_s = "SELECT WEAR_USER_ID,FROMPPG,DATE FROM andun_watch.d_bp_feature_model WHERE WEAR_USER_ID = '{}' AND DATE = '{}'".format(wear_user_id, date[0])
        else:
            return None

        res = self.ansql_read_mysql(sql_s)

        return res



    def ansql_bp_feature_with_date_range(self, wear_user_id, date_range):

        results = pd.DataFrame(columns=['WEAR_USER_ID','FROMPPG','DATE'])

        for dr in date_range:
            sql_s = "SELECT WEAR_USER_ID,FROMPPG,DATE FROM andun_watch.d_bp_feature_1 WHERE WEAR_USER_ID = '{}' AND DATE = '{}'".format(wear_user_id, dr)
        
            # if datetime.datetime.strptime(str(date), '%Y-%m-%d') < datetime.datetime.strptime("2020-04-26", '%Y-%m-%d'):
            #     sql_s = "SELECT FROMPPG FROM andun_watch.d_bp_feature WHERE WEAR_USER_ID = '{}' AND DATE in {}".format(wear_user_id, date)
            res = self.ansql_read_mysql(sql_s)
            if res.empty:
                continue
            else:
                # 拼接到 df_results
                results = pd.concat([results, res], ignore_index=True)
                
            time.sleep(0.5)
        return results


    def ansql_user_info(self, wear_user_id):
        sql_s = "SELECT BIRTHDAY, STATURE, WEIGHT, GENDER FROM andun_app.a_wear_user WHERE ID = '%s'" % wear_user_id
        res = self.ansql_read_mysql(sql_s)
        if res.empty:
            return None

        # 把BIRTHDAY计算成 年龄
        res['BIRTHDAY'] = self.cal_age_by_birthday(res['BIRTHDAY'].tolist()[0])
        # 
        res.rename(columns={'BIRTHDAY': 'Age', 'STATURE': 'Height', 'WEIGHT': 'Weight', 'GENDER': 'Gender'}, inplace=True)

        return res
        

    def ansql_user_age(self, wear_user_id):
        sql_s = "SELECT BIRTHDAY FROM andun_app.a_wear_user WHERE ID = '%s'" % wear_user_id
        res = self.ansql_read_mysql(sql_s)
        if res.empty:
            return None
        return self.cal_age_by_birthday(res['BIRTHDAY'][0])

    def ansql_user_height(self, wear_user_id):
        sql_s = "SELECT STATURE FROM andun_app.a_wear_user WHERE ID = '%s'" % wear_user_id
        res = self.ansql_read_mysql(sql_s)
        if res.empty:
            return None
        return int(res['STATURE'].values[0])

    def ansql_user_weight(self, wear_user_id):
        sql_s = "SELECT WEIGHT FROM andun_app.a_wear_user WHERE ID = '%s'" % wear_user_id
        res = self.ansql_read_mysql(sql_s)
        if res.empty:
            return None
        return int(res['WEIGHT'].values[0])

    def ansql_user_gender(self, wear_user_id):
        sql_s = "SELECT GENDER FROM andun_app.a_wear_user WHERE ID = '%s'" % wear_user_id
        res = self.ansql_read_mysql(sql_s)
        if res.empty:
            return None
        return int(res['GENDER'].values[0])

    def ansql_bp_true_label(self, wear_user_id):
        sql_s = 'SELECT gmt_create, sbp, dbp FROM andun_watch.c_bp_history WHERE wear_user_id="%s" AND status=0' % wear_user_id
        res = self.ansql_read_mysql(sql_s)
        if res.empty:
            return None
        return res

    def ansql_user_start_wear_date(self, wear_user_id):
        # CREATE_TIME 是 用户绑定日期, FIRST_COMMUNICATION_TIME 是手表首次上传数据的时间
        sql_s = "SELECT CREATE_TIME, FIRST_COMMUNICATION_TIME FROM andun_app.a_wear_user WHERE ID = '%s'" % wear_user_id
        res = self.ansql_read_mysql(sql_s)

        start_date = res['FIRST_COMMUNICATION_TIME'].tolist()[0]
        start_date = datetime.datetime.strftime(start_date, '%Y-%m-%d')

        return start_date

    def ansql_user_info_with_new_model(self, wear_user_id):
        sql_s = "SELECT * FROM andun_watch.d_users_bp_model WHERE wear_user_id = '%s'" % wear_user_id
        res = self.ansql_read_mysql(sql_s)
        # res = self.ansql_read_mysql_test_db(sql_s)
        return res


    def update_kefu_bp_history(self, wear_user_id, gmt_create):
        # 删除 客服记录表里的 一些记录数据
        sql_update = "UPDATE andun_cms.c_bp_history SET enabled=0 WHERE wear_user_id = '{}' AND gmt_create = '{}' ".format(wear_user_id, gmt_create)
        db = pymysql.connect(host=self._DB_HOST, user=self._DB_ACCOUNT, password=self._DB_PASSWORD, charset='utf8')
        cursor = db.cursor()
        try:
            cursor.execute(sql_update)
            db.commit()
        except Exception as e:
            print(e)
            print("修改 andun_cms.c_bp_history 发生错误...{}-{}".format(wear_user_id, gmt_create))
        db.close()

    # # 修改 d_users_bp_model 的 update_time值
    def update_d_users_bp_model(self, wear_user_id, update_time):
        sql_update = "UPDATE andun_watch.d_users_bp_model SET update_time='{}' WHERE wear_user_id = '{}'".format(update_time, wear_user_id)
        db = pymysql.connect(host=self._DB_HOST, user=self._DB_ACCOUNT, password=self._DB_PASSWORD, charset='utf8')
        cursor = db.cursor()
        try:
            cursor.execute(sql_update)
            db.commit()
        except Exception as e:
            print(e)
            print("修改 andun_cms.d_users_bp_model 发生错误...{}-{}".format(wear_user_id, update_time))
        db.close()


    # 新增 d_users_bp_model 的 记录
    def insert_d_users_bp_model(self, wear_user_id, create_time):
        sql_update = "INSERT INTO andun_watch.d_users_bp_model (wear_user_id, create_time, max_sbp, min_sbp, max_dbp, min_dbp, old_model_weight, kefu_model_weight, personal_model_weight,update_time) VALUES('{}', '{}', '{}', '{}', '{}','{}', '{}', '{}', '{}', '{}')".format(wear_user_id, create_time, 200,85,140,45,0.0,0.0,1.0, create_time)
        db = pymysql.connect(host=self._DB_HOST, user=self._DB_ACCOUNT, password=self._DB_PASSWORD, charset='utf8')
        cursor = db.cursor()
        try:
            cursor.execute(sql_update)
            db.commit()
        except Exception as e:
            print(e)
            print("添加 andun_cms.d_users_bp_model 发生错误...{}-{}".format(wear_user_id, create_time))
        db.close()

    # 删除 d_users_bp_model 的 记录
    def delete_d_users_bp_model(self, wear_user_id):
        sql_update = "DELETE FROM andun_watch.d_user_bp_model WHERE wear_user_id = '{}'".format(wear_user_id)
        db = pymysql.connect(host=self._DB_HOST, user=self._DB_ACCOUNT, password=self._DB_PASSWORD, charset='utf8')
        cursor = db.cursor()
        try:
            cursor.execute(sql_update)
            db.commit()
        except Exception as e:
            print(e)
            print("删除 andun_watch.d_users_bp_model 发生错误...{}".format(wear_user_id))
        db.close()

    # 删除 d_user_bp_nn_model_para 的 记录
    def delete_d_user_bp_nn_model_para(self, wear_user_id):
        sql_update = "DELETE FROM andun_watch.d_user_bp_nn_model_para WHERE wear_user_id = '{}'".format(wear_user_id)
        db = pymysql.connect(host=self._DB_HOST, user=self._DB_ACCOUNT, password=self._DB_PASSWORD, charset='utf8')
        cursor = db.cursor()
        try:
            cursor.execute(sql_update)
            db.commit()
        except Exception as e:
            print(e)
            print("删除 andun_watch.d_user_bp_nn_model_para 发生错误...{}".format(wear_user_id))
        db.close()

    