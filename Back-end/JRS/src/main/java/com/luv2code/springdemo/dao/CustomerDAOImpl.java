package com.luv2code.springdemo.dao;


import java.util.List;
import org.hibernate.Session;
import org.hibernate.SessionFactory;
import org.hibernate.query.Query;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Repository;
import com.luv2code.springdemo.entity.User;

@Repository
public class CustomerDAOImpl implements CustomerDAO {

	// need to inject the session factory
	@Autowired
	private SessionFactory sessionFactory;
			

	
	@Override
	public List<User> getUser() {
		Session currentSession = sessionFactory.getCurrentSession();
		Query<User> theQuery = currentSession.createQuery("from User order by id",User.class);									
		List<User> customers = theQuery.getResultList();
		return customers;
	}
	
	
	@Override
	public void saveUser(User user) {
		Session currentSession = sessionFactory.getCurrentSession();
		currentSession.saveOrUpdate(user);
	}
	
	
	

	@Override
	public User getUserById(int theId) {
		Session currentSession = sessionFactory.getCurrentSession();
		User theCustomer = currentSession.get(User.class, theId);
		return theCustomer;
	}
	

	
	
	



}











