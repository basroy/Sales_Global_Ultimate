from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField
from datetime import datetime

# import os

dbapp = Flask(__name__)
dbapp.config['SECRET_KEY'] = 'Impala!'
dbapp.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
dbapp.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///dbsrep.db'
db = SQLAlchemy(dbapp)


class People(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    srepname = db.Column(db.String(40), nullable=False) 
    title = db.Column(db.String(100), nullable=False)
    content = db.Column(db.String(100), nullable=False)
    completed = db.Column(db.Integer, nullable=False, default=1)
    date_created = db.Column(db.DateTime, nullable=False, default=datetime.utcnow())

    def __repr__(self):
        return 'people data ' + str(self.id) + self.srepname + self.title + self.content + self.date_created

class searchSRep(FlaskForm):
    srepname = StringField('srepname') 
    sreptitle = StringField('sreptitle')
    srepcontent = StringField('content')

userlist = [
    {
        'user': 'basroy',
        'theatre': ["APJC", "Americas", "EMEA", "WorldWide"],
        'overlay': ["DISTI", "WWW-CH", "CX", "VSCH"],
        'SalesMotion': 'yes'
    },
    {
        'user': 'meecah',
        'theatre': ["Americas", "EMEA", "WorldWide"],
        'overlay': ["DISTI", "CH", "CX", "VSCH"]
    }
    ]


@dbapp.route('/index_form', methods=['GET', 'POST'])
def index_form():
    if request.method == 'POST':
        srepn = request.form['srepname'] 
        srept = request.form['sreptitle']   
        srepc = request.form['srepcontent'] 
        new_rec = People(srepname=srepn, title=srept, content=srepc)
    
        try:
            db.session.add(new_rec)
            db.session.commit()
            return redirect('index_form')
        except:
            return 'An issue with adding data to Model people'

    else:
        dispall = People.query.order_by(People.date_created).all()
        for rec in dispall:
            print(rec.srepname)
            print(f"<id={rec.id}, srepname={rec.srepname}>")
        return  render_template('index_form.html', allsrep = dispall)
    #    return f'<h1> The user details: { dispall.content.data }</h1> '


#     if srch_form.validate_on_submit():
#         return '<h3> salesrep {}. has title as {}.'.format(srch_form.srepname.data, srch_form.sreptitle.data) 
#  
    srch_form = searchSRep() # used with WTFOrms  , in alignment with index_form.html having the form.csrf_token block
#    return render_template('index_form.html', form=srch_form)
   
# Add records to People
@dbapp.route('/<name>')
def index(name):
    #name = 'Bashobi'
    user = People(srepname=name, title='Sales Manager', content='Named Accounts  DIRECT')
    db.session.add(user)
    db.session.commit()
    
    return render_template('/index_local.html', users=userlist)
    #return '<h1>List of Salesreps</h1>'

# Issue Search in people
@dbapp.route('/search:<name>')
def get_srep(name):
    user = People.query.filter_by(srepname=name).first()

    return f'<h1> The user details: { user.content }</h1> '

if __name__ == "__main__":
   dbapp.run(debug=True)
   