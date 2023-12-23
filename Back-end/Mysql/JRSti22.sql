-- Create the database
CREATE DATABASE IF NOT EXISTS YourDatabaseName;
USE YourDatabaseName;

-- Create the User table
CREATE TABLE IF NOT EXISTS User (
    id INT AUTO_INCREMENT PRIMARY KEY,
    first_name VARCHAR(255) NOT NULL,
    last_name VARCHAR(255) NOT NULL,
    email VARCHAR(255) NOT NULL,
    address VARCHAR(255) NOT NULL,
    phone_number VARCHAR(15) NOT NULL,
    agile INT NOT NULL,
    Teamwork INT NOT NULL,
    knowledge INT NOT NULL,
    Speed INT NOT NULL,
    skillEnglish INT NOT NULL,
    attempt INT NOT NULL,
    idea INT NOT NULL,
    cleanCode INT NOT NULL,
    document INT NOT NULL,
    polite INT NOT NULL
);
