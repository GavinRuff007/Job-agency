import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import RegisterPage from './RegisterPage';
import { signUp } from 'next-auth/react'; // Mock NextAuth for API call

jest.mock('next-auth/react'); // Isolate NextAuth for testing

test('should handle successful registration', async () => {
  signUp.mockResolvedValueOnce({ error: null }); // Mock successful registration

  render(<RegisterPage />);

  const fullnameInput = screen.getByLabelText('نام و نام خانوادکی');
  const usernameInput = screen.getByLabelText('نام کاربری');
  const passwordInput = screen.getByLabelText('کلمه عبور');
  const submitButton = screen.getByRole('button', { name: /ثبت نام/i });

  userEvent.type(fullnameInput, 'John Doe');
  userEvent.type(usernameInput, 'johndoe');
  userEvent.type(passwordInput, 'password123');
  userEvent.click(submitButton);

  await waitFor(() => {
    expect(signUp).toHaveBeenCalledWith('credentials', {
      fullname: 'John Doe',
      username: 'johndoe',
      password: 'password123',
      redirect: false,
      callbackUrl: '/api/auth/signin', // Assuming redirect URL
    });
    expect(router.push).toHaveBeenCalledWith('/api/auth/signin');
  });
});

test('should handle failed registration', async () => {
  signUp.mockRejectedValueOnce(new Error('Invalid credentials'));

  render(<RegisterPage />);

  // ... (similar to previous test, but asserting error message)
});

test('should display error message for empty fields', async () => {
  render(<RegisterPage />);

  const submitButton = screen.getByRole('button', { name: /ثبت نام/i });
  userEvent.click(submitButton);

  await waitFor(() => {
    expect(screen.getByText('نام و نام خانوادگی الزامی است')).toBeInTheDocument();
    expect(screen.getByText('نام کاربری الزامی است')).toBeInTheDocument();
    expect(screen.getByText('کلمه عبور الزامی است')).toBeInTheDocument();
  });
});
