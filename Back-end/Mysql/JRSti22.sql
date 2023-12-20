-- Create the JRSti22 database
CREATE DATABASE IF NOT EXISTS JRSti22;
USE JRSti22;

-- Create the User table
CREATE TABLE IF NOT EXISTS User (
    id INT AUTO_INCREMENT PRIMARY KEY,
    first_name VARCHAR(255) NOT NULL,
    last_name VARCHAR(255) NOT NULL,
    email VARCHAR(255) NOT NULL,
    phone VARCHAR(15) NOT NULL,
    address VARCHAR(255) NOT NULL
);

-- Create the Score table
CREATE TABLE IF NOT EXISTS Score (
    idSeeker INT,
    Teamwork INT,
    Attempt INT,
    idea INT,
    Skill_English INT,
    Knowledge INT,
    Speed INT,
    Clean_code INT,
    Document INT,
    Agile INT,
    PRIMARY KEY (idSeeker),
    FOREIGN KEY (idSeeker) REFERENCES User(id)
);

-- Test Query for find
SELECT
    Score.Teamwork,
    Score.Attempt,
    Score.idea,
    Score.Skill_English,
    Score.Knowledge,
    Score.Speed,
    Score.Clean_code,
    Score.Document,
    Score.Agile
FROM
    Score
WHERE
    Score.idSeeker = 1;



