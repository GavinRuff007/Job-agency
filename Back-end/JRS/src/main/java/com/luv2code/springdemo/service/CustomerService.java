package com.luv2code.springdemo.service;

import java.util.List;
import com.luv2code.springdemo.entity.User;

public interface CustomerService {


	public void saveUser(User user);

	public User getUserById(int theId);

	public List<User> getUser();

	public void updateUser(User user);
	
}
