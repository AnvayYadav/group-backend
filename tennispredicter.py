import csv

with open("players.csv", mode = "w") as csvfile:
    fieldnames = ["_title", "_field", "_qualification",  "_pay"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow({"_title": " Software Engineer", "_field": "Computer Science", "_qualification": "Bachelors", "_pay": "37"})
    writer.writerow({"_title": " Software Engineer", "_field": "Computer Science", "_qualification": "Masters", "_pay": "45"})
    writer.writerow({"_title": " Software Engineer", "_field": "Computer Science", "_qualification": "pHd", "_pay": "56"})

    writer.writerow({"_title": "Data Scientist", "_field": "Computer Science", "_qualification": "Bachelors", "_pay": "32"})
    writer.writerow({"_title": "Data Scientist", "_field": "Computer Science", "_qualification": "Masters", "_pay": "41"})
    writer.writerow({"_title": "Data Scientist", "_field": "Computer Science", "_qualification": "pHd", "_pay": "49"})


    writer.writerow({"_title": "Mechanical Designer", "_field": "Engineering", "_qualification": "Bachelors", "_pay": "33"})
    writer.writerow({"_title": "Mechanical Designer", "_field": "Engineering", "_qualification": "Masters", "_pay": "42"})
    writer.writerow({"_title": "Mechanical Designer", "_field": "Engineering", "_qualification": "pHd", "_pay": "51"})

    writer.writerow({"_title": "Simulation Tester", "_field": "Engineering", "_qualification": "Bachelors", "_pay": "30"})
    writer.writerow({"_title": "Simulation Tester", "_field": "Engineering", "_qualification": "Masters", "_pay": "37"})
    writer.writerow({"_title": "Simulation Tester", "_field": "Engineering", "_qualification": "pHd", "_pay": "45"})


    writer.writerow({"_title": "Lawyer", "_field": "Law", "_qualification": "Bachelors", "_pay": "38"})
    writer.writerow({"_title": "Lawyer", "_field": "Law", "_qualification": "Masters", "_pay": "46"})
    writer.writerow({"_title": "Lawyer", "_field": "Law", "_qualification": "Masters ", "_pay": "55"})

    writer.writerow({"_title": "Judge", "_field": "Law", "_qualification": "Bachelors", "_pay": "35"})
    writer.writerow({"_title": "Judge", "_field": "Law", "_qualification": "Masters", "_pay": "48"})
    writer.writerow({"_title": "Judge", "_field": "Law", "_qualification": "Masters", "_pay": "62"})

    writer.writerow({"_title": "Doctor", "_field": "Medical", "_qualification": "Bachelors", "_pay": "30"})
    writer.writerow({"_title": "Doctor", "_field": "Medical", "_qualification": "Masters", "_pay": "30"})
    writer.writerow({"_title": "Doctor", "_field": "Medical", "_qualification": "pHd", "_pay": "30"})

    writer.writerow({"_title": "Nurse", "_field": "Medical", "_qualification": "Bachelors", "_pay": "30"})
    writer.writerow({"_title": "Nurse", "_field": "Medical", "_qualification": "Masters", "_pay": "30"})
    writer.writerow({"_title": "Nurse", "_field": "Medical", "_qualification": "pHd", "_pay": "30"})
    
    writer.writerow({"_title": "Surgeon", "_field": "Medical", "_qualification": "Bachelors", "_pay": "30"})
    writer.writerow({"_title": "Surgeon", "_field": "Medical", "_qualification": "Masters", "_pay": "30"})
    writer.writerow({"_title": "Surgeon", "_field": "Medical", "_qualification": "pHd", "_pay": "30"})
    
