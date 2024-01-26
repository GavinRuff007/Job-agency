package com.luv2code.springdemo.rest;

import java.util.List;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import com.luv2code.springdemo.entity.User;
import com.luv2code.springdemo.service.CustomerService;

@RestController
@RequestMapping("/api")
public class CustomerRestController {

	@Autowired
	private CustomerService customerService;
	
	@GetMapping("/allUser")
	public List<User> getUser() {
	    return customerService.getUser();
	}
	
	
	@PostMapping("/saveUser")
	public User addCustomer(@RequestBody User theC) {
		theC.setId(0);
		customerService.saveUser(theC);
		return theC;
	}
	
	
	@GetMapping("/User/{UserId}")
	public User getUserById(@PathVariable int UserId) {
		User the = customerService.getUserById(UserId);
		if(the == null) {
			throw new CustomerNotFoundException("Customer id Not found - "+ UserId);
		}
		return the;
	}
	
	
	@PutMapping("/updateUser")
	public User updateCustomer(@RequestBody User theC) {
		customerService.saveUser(theC);
		return theC;
	}
	
}