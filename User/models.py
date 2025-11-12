from flask_sqlalchemy import SQLAlchemy
import datetime

# Initialize the SQLAlchemy object
db = SQLAlchemy()

class QueryLog(db.Model):
    # This is the table name
    __tablename__ = 'query_log'

    # Define the columns
    id = db.Column(db.String(36), primary_key=True, unique=True) # Use the request_id from your app
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    query_text = db.Column(db.String(1000))
    classified_context = db.Column(db.String(50))
    response_text = db.Column(db.String(4000))
    was_ambiguous = db.Column(db.Boolean, default=False)
    was_no_docs = db.Column(db.Boolean, default=False)
    feedback = db.Column(db.SmallInteger, default=0) # 0=None, 1=Good, -1=Bad

    def __repr__(self):
        return f'<QueryLog {self.id}: {self.query_text[:50]}>'