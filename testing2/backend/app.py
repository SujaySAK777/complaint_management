import datetime
from flask import Flask, render_template, request, redirect, url_for, session , flash
from flask_sqlalchemy import SQLAlchemy
from nlp import preprocess_text, categorize_complaint, assign_priority,  analyze_sentiment
from datetime import datetime

import nltk

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('vader_lexicon')

app = Flask(__name__)

# Set up the database URI
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://username:password@localhost/database_name'  # Replace with actual credentials
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'your_secret_key_here'  # Update with a secure secret key
db = SQLAlchemy(app)

# Define models for the database
class Citizen(db.Model):
    __tablename__ = 'citizens'

    citizen_id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    contact_number = db.Column(db.String(20), nullable=False)
    email = db.Column(db.String(100), nullable=False, unique=True)
    address = db.Column(db.String(200), nullable=True)
    gender = db.Column(db.String(10), nullable=True)
    password = db.Column(db.String(100), nullable=True)
    complaints = db.relationship('Complaint', backref='citizen', lazy=True)

class Complaint(db.Model):
    __tablename__ = 'complaints'

    complaint_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    citizen_id = db.Column(db.Integer, db.ForeignKey('citizens.citizen_id'), nullable=False)
    category = db.Column(db.String(50))
    description = db.Column(db.Text, nullable=False)
    department_id = db.Column(db.Integer, db.ForeignKey('departments.department_id'), nullable=False)
    priority = db.Column(db.Enum('LOW', 'MEDIUM', 'HIGH'), nullable=False)
    date_submitted = db.Column(db.Date, nullable=True)

class Department(db.Model):
    __tablename__ = 'departments'

    department_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(50), nullable=False)
    contact_person = db.Column(db.String(50))
    contact_number = db.Column(db.String(15), unique=True)
    email = db.Column(db.String(50), unique=True)
    address = db.Column(db.String(100))
    password = db.Column(db.String(100))

class ComplaintLog(db.Model):
    __tablename__ = 'complaint_log'

    log_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    complaint_id = db.Column(db.Integer, db.ForeignKey('complaints.complaint_id'), nullable=False)
    status = db.Column(db.Enum('In-progress', 'Resolved', 'Not resolved'), nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())
    remarks = db.Column(db.Text)

    # Relationship to link logs to the complaint
    complaint = db.relationship('Complaint', backref=db.backref('logs', cascade='all, delete-orphan'))

# Home route
@app.route('/')
def index():
    return render_template('index.html')


# Citizen Registration Route
@app.route('/citizen-register', methods=['GET', 'POST'])
def citizen_register():
    if request.method == 'POST':
        name = request.form['name']
        contact_number = request.form['contact_number']
        email = request.form['email']
        address = request.form['address']
        gender = request.form['gender']
        password = request.form['password']  # Capture password from the form
        
        # Create new citizen record
        new_citizen = Citizen(
            name=name, 
            contact_number=contact_number, 
            email=email, 
            address=address, 
            gender=gender, 
            password=password  # Save password directly
        )
        db.session.add(new_citizen)
        db.session.commit()

        return redirect(url_for('citizen_login'))

    return render_template('register.html')

# Citizen Login Route
@app.route('/citizen-login', methods=['GET', 'POST'])
def citizen_login():
    error = None
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']  # Capture password from the form
        
        # Fetch citizen by email and password
        citizen = Citizen.query.filter_by(email=email, password=password).first()
        
        if citizen:  # Validate email and password
            session['citizen_id'] = citizen.citizen_id  # Store citizen ID in session
            return redirect(url_for('citizen_dashboard'))
        else:
            error = 'Invalid email or password. Please try again.'

    return render_template('login.html', error=error)

# Complaint Submission Route
@app.route('/register-complaint', methods=['GET', 'POST'])
def register_complaint():
    if 'citizen_id' not in session:
        return redirect(url_for('citizen_login'))

    citizen_id = session['citizen_id']
    citizen = Citizen.query.get(citizen_id)  # Get the logged-in citizen's data

    if request.method == 'POST':
        description = request.form['description']
        
        # NLP processing
        processed_text = preprocess_text(description)
        category, department_id = categorize_complaint(processed_text)  # Adjust to return both category and department_id
        priority = assign_priority(processed_text)

        # Map category to department_id (already handled in categorize_complaint)
        # department_id is already returned by categorize_complaint, so no need to use the category_to_department dict here

        # Create the new complaint record
        new_complaint = Complaint(
            citizen_id=citizen_id,
            category=category,
            description=description,
            department_id=department_id,
            priority=priority,
            date_submitted=db.func.current_date(),
        )
        
        db.session.add(new_complaint)
        db.session.commit()

        return redirect(url_for('citizen_dashboard'))

    return render_template('register_complaint.html', citizen=citizen)


# Logout Route
@app.route('/logout')
def logout():
    session.pop('citizen_id', None)
    session.pop('department_id', None)
    return redirect(url_for('index'))

# Department Registration Route
@app.route('/department-register', methods=['GET', 'POST'])
def department_register():
    if request.method == 'POST':
        name = request.form['name']
        contact_person = request.form['contact_person']
        contact_number = request.form['contact_number']
        email = request.form['email']
        address = request.form['address']
        password = request.form['password']  # Password entered by the user

        # Create a new Department record
        new_department = Department(
            name=name,
            contact_person=contact_person,
            contact_number=contact_number,
            email=email,
            address=address,
            password=password  # Store the plain text password (not recommended for production)
        )
        db.session.add(new_department)
        db.session.commit()

        return redirect(url_for('department_login'))

    return render_template('department_register.html')

# Department Login Route
@app.route('/department-login', methods=['GET', 'POST'])
def department_login():
    error = None
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']  # Password entered by the user

        # Query the database for a matching email and password
        department = Department.query.filter_by(email=email, password=password).first()
        if department:
            # Store department_id in session
            session['department_id'] = department.department_id
            return redirect(url_for('department_dashboard'))
        else:
            error = 'Invalid login credentials. Please try again.'

    return render_template('department_login.html', error=error)




# Citizen Dashboard Route
@app.route('/citizen-dashboard')
def citizen_dashboard():
    if 'citizen_id' not in session:
        return redirect(url_for('citizen_login'))

    citizen_id = session['citizen_id']
    citizen = Citizen.query.get(citizen_id)
    complaints = Complaint.query.filter_by(citizen_id=citizen_id).all()
    return render_template('citizen_dashboard.html', citizen=citizen, complaints=complaints)

# Department Dashboard Route

@app.route('/department-dashboard', methods=['GET', 'POST'])
def department_dashboard():
    if 'department_id' not in session:
        return redirect(url_for('department_login'))

    department_id = session['department_id']
    selected_status = request.args.get('status', 'all')  # Filter by status (optional)

    # Mapping of department ID to name
    department_names = {
        1: "Sanitation",
        2: "Water",
        3: "Infrastructure",
        4: "Public Safety",
        5: "General"
    }

    department_name = department_names.get(department_id, "Unknown Department")

    # Query complaints for the department, sorting by priority
    query = Complaint.query.filter_by(department_id=department_id)

    if selected_status != 'all':
        query = query.join(ComplaintLog).filter(ComplaintLog.status == selected_status)

    # Sort complaints by priority (assuming priority is an integer or comparable field)
    complaints = query.order_by(Complaint.priority.asc()).all()  # Sort by priority in ascending order

    # Add latest status and remarks for each complaint
    for complaint in complaints:
        latest_log = ComplaintLog.query.filter_by(complaint_id=complaint.complaint_id).order_by(ComplaintLog.timestamp.desc()).first()
        complaint.latest_status = latest_log.status if latest_log else "Pending"
        complaint.latest_remarks = latest_log.remarks if latest_log else "No remarks yet"

    if request.method == 'POST':
        for complaint in complaints:
            complaint_id = request.form.get(f'complaint_id_{complaint.complaint_id}')
            status = request.form.get(f'status_{complaint.complaint_id}')
            remarks = request.form.get(f'remarks_{complaint.complaint_id}')

            if complaint_id:
                # Create a new entry in ComplaintLog for status and remarks
                log_entry = ComplaintLog(
                    complaint_id=complaint_id,
                    status=status,
                    remarks=remarks,
                    timestamp=datetime.now()
                )
                db.session.add(log_entry)
                db.session.commit()

                # Fetch the updated complaint and its latest log entry again after the update
                complaint = Complaint.query.get(complaint_id)
                latest_log = ComplaintLog.query.filter_by(complaint_id=complaint.complaint_id).order_by(ComplaintLog.timestamp.desc()).first()

                # Update the status and remarks to the complaint object for frontend rendering
                complaint.latest_status = latest_log.status if latest_log else "Pending"
                complaint.latest_remarks = latest_log.remarks if latest_log else "No remarks yet"

    return render_template('department_dashboard.html', complaints=complaints, selected_status=selected_status, department_name=department_name)

# View Complaints Route
@app.route('/view-complaint/<int:complaint_id>', methods=['GET'])
def view_complaint(complaint_id):
    complaint = Complaint.query.get(complaint_id)

    if not complaint:
        return "Complaint not found", 404

    # Fetch the latest log entry based on the timestamp for the specific complaint
    latest_log = ComplaintLog.query.filter_by(complaint_id=complaint_id).order_by(ComplaintLog.timestamp.desc()).first()

    # Fetch all logs for the complaint
    logs = ComplaintLog.query.filter_by(complaint_id=complaint_id).order_by(ComplaintLog.timestamp.desc()).all()

    return render_template('view_complaint.html', complaint=complaint, latest_log=latest_log, logs=logs)


# Update Complaint Status Route
@app.route('/update-complaint-status/<int:complaint_id>', methods=['GET', 'POST'])
def update_complaint_status(complaint_id):
    if 'department_id' not in session:
        flash('You need to log in first.', 'warning')
        return redirect(url_for('department_login'))

    # Fetch the complaint for the logged-in department
    complaint = Complaint.query.get(complaint_id)
    if not complaint or complaint.department_id != session['department_id']:
        flash('Unauthorized access.', 'danger')
        return redirect(url_for('department_dashboard'))

    if request.method == 'POST':
        status = request.form.get('status')
        remarks = request.form.get('remarks')

        if not status or not remarks:
            flash('Please provide both status and remarks.', 'warning')
            return redirect(url_for('update_complaint_status', complaint_id=complaint_id))

        # Add a new log entry to the ComplaintLog table
        log = ComplaintLog(
            complaint_id=complaint_id,
            status=status,
            remarks=remarks,
            timestamp=datetime.now()  # Ensure timestamp is recorded
        )
        db.session.add(log)
        db.session.commit()  # Commit changes to the database

        flash('Complaint updated successfully.', 'success')
        return redirect(url_for('department_dashboard'))

    return render_template('update_complaint.html', complaint=complaint)


# Run the application
if __name__ == "__main__":
    app.run(debug=True)
