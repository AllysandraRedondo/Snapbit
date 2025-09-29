from flask import Flask, render_template, request, redirect, url_for
from flask_mail import Mail, Message
import os

app = Flask(__name__)

app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USERNAME'] = "xxxxxxxxx@gmail.com"        
app.config['MAIL_PASSWORD'] = "xxxxxxxxxxxx"
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False

mail = Mail(app)

@app.route("/")
def index():
    return render_template("Email_Page.html")

@app.route("/submit", methods=["POST"]) #Route ng function and method use for email submit function
def submit():
    if request.method == "POST":
        email = request.form["email"]

        msg = Message( #Message function for the email
            subject="SnapBits Photo",
            sender=app.config['MAIL_USERNAME'], 
            recipients=[email]
        )
        msg.body = "Hi! This is your photo. Please download it within 3 days."

        try:
            mail.send(msg)
            print("✅ Email sent successfully!")
        except Exception as e:
            print("Error Mhiema:", e)

        return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)
