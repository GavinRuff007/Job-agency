package com.luv2code.springdemo.dao;

import java.util.List;
import com.luv2code.springdemo.entity.User;

public interface CustomerDAO {


	public void saveUser(User theCustomer);

	public User getUserById(int theId);

	public List<User> getUser();
	
}
