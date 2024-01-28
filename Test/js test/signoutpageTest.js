
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import SignOutButton from './SignOutButton';
import { signOut } from 'next-auth/react'; // Mock NextAuth for API call

jest.mock('next-auth/react'); // Isolate NextAuth for testing

test('should render the button with correct text', () => {
  render(<SignOutButton />);

  const button = screen.getByRole('button', { name: /آره مطمئنم/i });
  expect(button).toBeInTheDocument();
});

test('should call signOut function when button is clicked', async () => {
  render(<SignOutButton />);

  const button = screen.getByRole('button', { name: /آره مطمئنم/i });
  userEvent.click(button);

  await waitFor(() => {
    expect(signOut).toHaveBeenCalledWith({ callbackUrl: '/' });
  });
});

// Requires a testing environment that supports browser-like behavior
test('should redirect to the callback URL after sign-out', async () => {
  render(<SignOutButton />);

  const button = screen.getByRole('button', { name: /آره مطمئنم/i });
  userEvent.click(button);

  await waitFor(() => {
    expect(window.location.href).toBe('/'); // Assuming redirect URL
  });
});
