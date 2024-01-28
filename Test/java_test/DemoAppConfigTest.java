package com.luv2code.springdemo.config;

import static org.junit.Assert.assertEquals;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.junit4.SpringRunner;

@RunWith(SpringRunner.class)
@SpringBootTest
public class DemoAppConfigTest {

    @Autowired
    private DemoAppConfig demoAppConfig;

    @Test
    public void testMyDataSource() {
        DataSource dataSource = demoAppConfig.myDataSource();
        assertEquals("com.mysql.jdbc.Driver", dataSource.getDriverClassName());
        assertEquals("jdbc:mysql://localhost:3306/my_database", dataSource.getUrl());
        assertEquals("root", dataSource.getUsername());
        assertEquals("password", dataSource.getPassword());
    }

    @Test
    public void testGetHibernateProperties() {
        Properties hibernateProperties = demoAppConfig.getHibernateProperties();
        assertEquals("org.hibernate.dialect.MySQLDialect", hibernateProperties.getProperty("hibernate.dialect"));
        assertEquals("true", hibernateProperties.getProperty("hibernate.show_sql"));
    }

    @Test
    public void testSessionFactory() {
        LocalSessionFactoryBean sessionFactory = demoAppConfig.sessionFactory();
        assertEquals("com.mysql.jdbc.Driver", sessionFactory.getDataSource().getDriverClassName());
        assertEquals("jdbc:mysql://localhost:3306/my_database", sessionFactory.getDataSource().getUrl());
        assertEquals("root", sessionFactory.getDataSource().getUsername());
        assertEquals("password", sessionFactory.getDataSource().getPassword());
        assertEquals("org.hibernate.dialect.MySQLDialect", sessionFactory.getHibernateProperties().getProperty("hibernate.dialect"));
        assertEquals("true", sessionFactory.getHibernateProperties().getProperty("hibernate.show_sql"));
    }

    @Test
    public void testTransactionManager() {
        HibernateTransactionManager transactionManager = demoAppConfig.transactionManager(demoAppConfig.sessionFactory());
        assertEquals(demoAppConfig.sessionFactory(), transactionManager.getSessionFactory());
    }
}
