package com.luv2code.springdemo.service;

import java.util.List;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import com.luv2code.springdemo.dao.CustomerDAO;
import com.luv2code.springdemo.entity.User;

@Service
public class CustomerServiceImpl implements CustomerService {

	@Autowired
	private CustomerDAO customerDAO;
	
	
	@Override
	@Transactional
	public List<User> getUser() {
	    return customerDAO.getUser();
	}


	@Override
	@Transactional
	public void saveUser(User user) {
		customerDAO.saveUser(user);
	}
	
	
	@Override
	@Transactional
	public User getUserById(int theId) {
		return customerDAO.getUserById(theId);
	}
	
	@Override
	@Transactional
	public void updateUser(User user) {
		customerDAO.saveUser(user);
	}

}