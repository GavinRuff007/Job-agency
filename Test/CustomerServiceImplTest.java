
package com.luv2code.springdemo.service;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import java.util.Arrays;
import java.util.List;

import org.junit.Test;

public class CustomerServiceImplTest {

    @Test
    public void testGetUser() {
        CustomerServiceImpl customerService = new CustomerServiceImpl();
        CustomerDAO customerDAO = mock(CustomerDAO.class);
        customerService.customerDAO = customerDAO;

        List<User> expectedUsers = Arrays.asList(new User("John Doe", "johndoe@example.com"), new User("Jane Doe", "janedoe@example.com"));
        when(customerDAO.getUser()).thenReturn(expectedUsers);

        List<User> actualUsers = customerService.getUser();

        assertEquals(expectedUsers, actualUsers);
    }

    @Test
    public void testSaveUser() {
        CustomerServiceImpl customerService = new CustomerServiceImpl();
        CustomerDAO customerDAO = mock(CustomerDAO.class);
        customerService.customerDAO = customerDAO;

        User user = new User("John Doe", "johndoe@example.com");

        when(customerDAO.saveUser(user)).thenReturn(

    }
}