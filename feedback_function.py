import mysql.connector 
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

def get_db(): 
    return mysql.connector.connect(
        host='localhost', 
        user='root',
        password= '',
        database='snapbits_db'
    )

@app.route ("/feedback")
def feedback():
    return render_template("Feedback.html")


@app.route("/submit-feedback", methods=["POST"])
def submit_feedback():
    data = request.get_json() 
    rating = data['rating']

#def save_feedback(rating):
    try: 
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO feedback_tbl (rating) VALUES (%s)", 
            (rating,)
        )
        conn.commit()
        cursor.close()
        conn.close()

        print(f"Rating is saved: {rating} stars")
        return jsonify({"message": "Thank you for your feedback!"})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Failed to save the rating"})
    
if __name__ == "__main__":
    app.run(debug=True)
    