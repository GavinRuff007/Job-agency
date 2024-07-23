package com.luv2code.springdemo.entity;

import javax.persistence.*;

@Entity
@Table(name = "User")
public class User {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "id")
    private int id;

    @Column(name = "first_name")
    private String firstName;

    @Column(name = "last_name")
    private String lastName;

    @Column(name = "email")
    private String email;

    @Column(name = "address")
    private String address;

    @Column(name = "phone_number")
    private String phoneNumber;

    @Column(name = "agile")
    private int agile;

    @Column(name = "Teamwork")
    private int teamwork;

    @Column(name = "knowledge")
    private int knowledge;

    @Column(name = "Speed")
    private int speed;

    @Column(name = "skillEnglish")
    private int skillEnglish;

    @Column(name = "attempt")
    private int attempt;

    @Column(name = "idea")
    private int idea;

    @Column(name = "cleanCode")
    private int cleanCode;

    @Column(name = "document")
    private int document;

    @Column(name = "polite")
    private int polite;


    public User() {
    }

    public int getId() {
        return id;
    }

    public void setId(int id) {
        this.id = id;
    }

    public String getFirstName() {
        return firstName;
    }

    public void setFirstName(String firstName) {
        this.firstName = firstName;
    }

    public String getLastName() {
        return lastName;
    }

    public void setLastName(String lastName) {
        this.lastName = lastName;
    }

    public String getEmail() {
        return email;
    }

    public void setEmail(String email) {
        this.email = email;
    }

    public String getAddress() {
        return address;
    }

    public void setAddress(String address) {
        this.address = address;
    }

    public String getPhoneNumber() {
        return phoneNumber;
    }

    public void setPhoneNumber(String phoneNumber) {
        this.phoneNumber = phoneNumber;
    }

    public int getAgile() {
        return agile;
    }

    public void setAgile(int agile) {
        this.agile = agile;
    }

    public int getTeamwork() {
        return teamwork;
    }

    public void setTeamwork(int teamwork) {
        this.teamwork = teamwork;
    }

    public int getKnowledge() {
        return knowledge;
    }

    public void setKnowledge(int knowledge) {
        this.knowledge = knowledge;
    }

    public int getSpeed() {
        return speed;
    }

    public void setSpeed(int speed) {
        this.speed = speed;
    }

    public int getSkillEnglish() {
        return skillEnglish;
    }

    public void setSkillEnglish(int skillEnglish) {
        this.skillEnglish = skillEnglish;
    }

    public int getAttempt() {
        return attempt;
    }

    public void setAttempt(int attempt) {
        this.attempt = attempt;
    }

    public int getIdea() {
        return idea;
    }

    public void setIdea(int idea) {
        this.idea = idea;
    }

    public int getCleanCode() {
        return cleanCode;
    }

    public void setCleanCode(int cleanCode) {
        this.cleanCode = cleanCode;
    }

    public int getDocument() {
        return document;
    }

    public void setDocument(int document) {
        this.document = document;
    }

    public int getPolite() {
        return polite;
    }

    public void setPolite(int polite) {
        this.polite = polite;
    }



    @Override
    public String toString() {
        return "User{" +
                "id=" + id +
                ", firstName='" + firstName + '\'' +
                ", lastName='" + lastName + '\'' +
                ", email='" + email + '\'' +
                ", address='" + address + '\'' +
                ", phoneNumber='" + phoneNumber + '\'' +
                ", agile=" + agile +
                ", teamwork=" + teamwork +
                ", knowledge=" + knowledge +
                ", speed=" + speed +
                ", skillEnglish=" + skillEnglish +
                ", attempt=" + attempt +
                ", idea=" + idea +
                ", cleanCode=" + cleanCode +
                ", document=" + document +
                ", polite=" + polite +

                '}';
    }
}
