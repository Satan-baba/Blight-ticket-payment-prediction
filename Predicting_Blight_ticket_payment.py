from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score, recall_score
import pandas as pd
import numpy as np
def blight_model():
    df_train = pd.read_csv('train.csv', encoding = "ISO-8859-1")
    df_test = pd.read_csv('test.csv', encoding = "ISO-8859-1")
    add = pd.read_csv('addresses.csv')
    latlon = pd.read_csv('latlons.csv')
    df_train = (latlon.set_index('address').join(add.set_index('address')).set_index('ticket_id')).join(df_train.set_index('ticket_id'),how = 'right')
    df_test = (latlon.set_index('address').join(add.set_index('address')).set_index('ticket_id')).join(df_test.set_index('ticket_id'), how = 'right')
    df_train = df_train[np.isfinite(df_train['compliance'])]
    features_to_drop_train = ['payment_amount','payment_date', 'payment_status','balance_due','collection_status','compliance_detail']
    df_train.drop(features_to_drop_train, axis = 1,inplace = True)
    import datetime
    def t_gap(hearing_date, ticket_issue_date):
        if not hearing_date or type(hearing_date)!=str:return 73
        ticket_issue = datetime.datetime.strptime(ticket_issue_date, '%Y-%m-%d %H:%M:%S')
        hearing_date_1 = datetime.datetime.strptime(hearing_date, '%Y-%m-%d %H:%M:%S')
        gap = hearing_date_1 - ticket_issue
        return gap.days
    df_train['days'] = df_train.apply(lambda x:t_gap(x['hearing_date'], x['ticket_issued_date']), axis = 1)
    df_test['days'] = df_test.apply(lambda x:t_gap(x['hearing_date'], x['ticket_issued_date']), axis = 1)
    features_to_bin = ['agency_name', 'state', 'disposition']
    df_train = pd.get_dummies(df_train, columns = features_to_bin)
    df_test = pd.get_dummies(df_test, columns = features_to_bin) 
    df_train.drop(df_train.columns[3:25].tolist(), axis = 1,inplace = True)
    df_test.drop(df_test.columns[2:25].tolist(), axis = 1, inplace = True)
    df_train_x = df_train.drop(['compliance'], axis = 1)
    df_train_y = df_train['compliance']
    df_train_x.drop(['inspector_name'], axis = 1, inplace = True)
    df_train_x['lat'].fillna(method = 'pad', inplace = True)
    df_train_x['lon'].fillna(method = 'pad', inplace = True)
    df_train_x1 = df_train[['disposition_Responsible (Fine Waived) by Deter','disposition_Responsible by Admission',
                       'disposition_Responsible by Default', 'disposition_Responsible by Determination']]
    df_test['lat'].fillna(method = 'pad', inplace = True)
    df_test['lon'].fillna(method = 'pad', inplace = True)
    for i in df_train_x.columns:
        for j in set(df_train_x.columns.tolist()) ^ set(df_test.columns.tolist()):
            if i == j:
                df_train_x.drop([i], axis = 1, inplace = True)
    for i in df_test.columns:
        for j in set(df_train_x.columns.tolist()) ^ set(df_test.columns.tolist()):
            if i == j:
                df_test.drop([i], axis = 1, inplace = True)
    
    clf = RandomForestClassifier(n_estimators = 25, max_depth = 15, max_features = None, random_state = 3)

    clf1 = clf.fit(df_train_x, df_train_y)
    y_pred = clf1.predict_proba(df_test)
    df_test['compliance'] = y_pred[:,1]

    return df_test['compliance']


