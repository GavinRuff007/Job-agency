import React from 'react';
import { render, screen } from '@testing-library/react';
import RootLayout from './layout';

// Mock localFont to simulate font loading
jest.mock('next/font/local', () => ({
  localFont: () => ({ className: 'mock-font-class' }),
}));

describe('RootLayout', () => {
  it('renders the correct HTML structure', () => {
    render(<RootLayout />);

    const html = screen.getByRole('html');
    expect(html).toHaveAttribute('lang', 'fa-IR');
    expect(html).toHaveAttribute('dir', 'rtl');
    expect(html).toHaveClass(/vazir/);

    const body = screen.getByRole('body');
    expect(body).toBeInTheDocument();

    const authProvider = screen.getByTestId('auth-provider');
    expect(authProvider).toBeInTheDocument();
  });

  it('loads the Vazir font', () => {
    render(<RootLayout />);

    const html = screen.getByRole('html');
    expect(html).toHaveClass('mock-font-class');
  });

  it('wraps children with AuthProvider', () => {
    const children = <div data-testid="child-content" />;
    render(<RootLayout>{children}</RootLayout>);

    const authProvider = screen.getByTestId('auth-provider');
    const childContent = screen.getByTestId('child-content');
    expect(childContent).toBeWithin(authProvider);
  });
});