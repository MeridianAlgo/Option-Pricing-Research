"""
Test Suite for Option Pricing Model
===================================

Comprehensive tests for the OptionPricing class and all pricing models.
"""

import unittest
import numpy as np
from option_pricing import OptionPricing


class TestOptionPricing(unittest.TestCase):
    """
    Test cases for the OptionPricing class
    """
    
    def setUp(self):
        """
        Set up test fixtures with common option parameters
        """
        self.S = 100.0  # Current stock price
        self.K = 105.0  # Strike price
        self.T = 0.25   # Time to expiration
        self.r = 0.05   # Risk-free rate
        self.sigma = 0.2  # Volatility
        
        self.call_option = OptionPricing(self.S, self.K, self.T, self.r, self.sigma, 'call')
        self.put_option = OptionPricing(self.S, self.K, self.T, self.r, self.sigma, 'put')
    
    def test_initialization(self):
        """
        Test proper initialization of OptionPricing objects
        """
        # Test call option initialization
        self.assertEqual(self.call_option.S, self.S)
        self.assertEqual(self.call_option.K, self.K)
        self.assertEqual(self.call_option.T, self.T)
        self.assertEqual(self.call_option.r, self.r)
        self.assertEqual(self.call_option.sigma, self.sigma)
        self.assertEqual(self.call_option.option_type, 'call')
        
        # Test put option initialization
        self.assertEqual(self.put_option.option_type, 'put')
        
        # Test invalid option type
        with self.assertRaises(ValueError):
            OptionPricing(self.S, self.K, self.T, self.r, self.sigma, 'invalid')
    
    def test_black_scholes_call(self):
        """
        Test Black-Scholes pricing for call options
        """
        price, delta, gamma, theta, vega = self.call_option.black_scholes()
        
        # Check that all values are positive (for this OTM call)
        self.assertGreater(price, 0)
        self.assertGreater(delta, 0)
        self.assertGreater(gamma, 0)
        self.assertLess(theta, 0)  # Theta should be negative (time decay)
        self.assertGreater(vega, 0)
        
        # Check that price is reasonable (less than stock price)
        self.assertLess(price, self.S)
        
        # Check that delta is between 0 and 1 for call
        self.assertGreaterEqual(delta, 0)
        self.assertLessEqual(delta, 1)
    
    def test_black_scholes_put(self):
        """
        Test Black-Scholes pricing for put options
        """
        price, delta, gamma, theta, vega = self.put_option.black_scholes()
        
        # Check that all values are positive (for this OTM put)
        self.assertGreater(price, 0)
        self.assertLess(delta, 0)  # Delta should be negative for put
        self.assertGreater(gamma, 0)
        self.assertLess(theta, 0)  # Theta should be negative (time decay)
        self.assertGreater(vega, 0)
        
        # Check that price is reasonable (less than strike price)
        self.assertLess(price, self.K)
        
        # Check that delta is between -1 and 0 for put
        self.assertGreaterEqual(delta, -1)
        self.assertLessEqual(delta, 0)
    
    def test_put_call_parity(self):
        """
        Test put-call parity relationship
        """
        call_price, _, _, _, _ = self.call_option.black_scholes()
        put_price, _, _, _, _ = self.put_option.black_scholes()
        
        # Put-call parity: C - P = S - K*e^(-rT)
        lhs = call_price - put_price
        rhs = self.S - self.K * np.exp(-self.r * self.T)
        
        # Should be approximately equal (within numerical precision)
        self.assertAlmostEqual(lhs, rhs, places=10)
    
    def test_binomial_model(self):
        """
        Test binomial model pricing
        """
        # Test with different number of steps
        for steps in [10, 50, 100]:
            price, stock_prices, option_values = self.call_option.binomial(steps)
            
            # Check that price is positive
            self.assertGreater(price, 0)
            
            # Check that stock prices matrix has correct dimensions
            self.assertEqual(stock_prices.shape, (steps + 1, steps + 1))
            self.assertEqual(option_values.shape, (steps + 1, steps + 1))
            
            # Check that stock prices are positive
            self.assertTrue(np.all(stock_prices >= 0))
            
            # Check that option values are non-negative
            self.assertTrue(np.all(option_values >= 0))
    
    def test_monte_carlo_model(self):
        """
        Test Monte Carlo model pricing
        """
        # Test with different number of simulations
        for sims in [1000, 10000, 100000]:
            price, std_error, payoffs = self.call_option.monte_carlo(sims)
            
            # Check that price is positive
            self.assertGreater(price, 0)
            
            # Check that standard error is positive
            self.assertGreater(std_error, 0)
            
            # Check that payoffs array has correct length
            self.assertEqual(len(payoffs), sims)
            
            # Check that all payoffs are non-negative
            self.assertTrue(np.all(payoffs >= 0))
    
    def test_model_convergence(self):
        """
        Test that binomial and Monte Carlo models converge to Black-Scholes
        """
        # Get Black-Scholes price as reference
        bs_price, _, _, _, _ = self.call_option.black_scholes()
        
        # Test binomial convergence
        bin_price_100, _, _ = self.call_option.binomial(100)
        bin_price_1000, _, _ = self.call_option.binomial(1000)
        
        # Higher step count should be closer to Black-Scholes
        error_100 = abs(bin_price_100 - bs_price)
        error_1000 = abs(bin_price_1000 - bs_price)
        
        self.assertLess(error_1000, error_100)
        
        # Test Monte Carlo convergence
        mc_price_1000, _, _ = self.call_option.monte_carlo(1000)
        mc_price_100000, _, _ = self.call_option.monte_carlo(100000)
        
        # Higher simulation count should be closer to Black-Scholes
        error_mc_1000 = abs(mc_price_1000 - bs_price)
        error_mc_100000 = abs(mc_price_100000 - bs_price)
        
        # Note: Monte Carlo has randomness, so we check that error is reasonable
        self.assertLess(error_mc_100000, 0.1)  # Should be within 10 cents
    
    def test_atm_options(self):
        """
        Test at-the-money options (S = K)
        """
        atm_call = OptionPricing(self.S, self.S, self.T, self.r, self.sigma, 'call')
        atm_put = OptionPricing(self.S, self.S, self.T, self.r, self.sigma, 'put')
        
        call_price, call_delta, _, _, _ = atm_call.black_scholes()
        put_price, put_delta, _, _, _ = atm_put.black_scholes()
        
        # For ATM options, call and put should have similar prices
        # Due to put-call parity: C - P = S - K*e^(-rT), for ATM: C - P = S*(1 - e^(-rT))
        expected_diff = self.S * (1 - np.exp(-self.r * self.T))
        actual_diff = call_price - put_price
        self.assertAlmostEqual(actual_diff, expected_diff, places=2)
        
        # Call delta should be around 0.5, put delta around -0.5
        self.assertAlmostEqual(call_delta, 0.5, places=0)
        self.assertAlmostEqual(put_delta, -0.5, places=0)
    
    def test_itm_options(self):
        """
        Test in-the-money options
        """
        # ITM call: S > K
        itm_call = OptionPricing(110, 100, self.T, self.r, self.sigma, 'call')
        call_price, call_delta, _, _, _ = itm_call.black_scholes()
        
        # ITM call should have high delta (close to 1)
        self.assertGreater(call_delta, 0.7)
        
        # ITM put: S < K
        itm_put = OptionPricing(90, 100, self.T, self.r, self.sigma, 'put')
        put_price, put_delta, _, _, _ = itm_put.black_scholes()
        
        # ITM put should have high negative delta (close to -1)
        self.assertLess(put_delta, -0.7)
    
    def test_otm_options(self):
        """
        Test out-of-the-money options
        """
        # OTM call: S < K
        otm_call = OptionPricing(90, 100, self.T, self.r, self.sigma, 'call')
        call_price, call_delta, _, _, _ = otm_call.black_scholes()
        
        # OTM call should have low delta (close to 0)
        self.assertLess(call_delta, 0.3)
        
        # OTM put: S > K
        otm_put = OptionPricing(110, 100, self.T, self.r, self.sigma, 'put')
        put_price, put_delta, _, _, _ = otm_put.black_scholes()
        
        # OTM put should have low negative delta (close to 0)
        self.assertGreater(put_delta, -0.3)
    
    def test_time_decay(self):
        """
        Test that option prices decrease with time (theta negative)
        """
        # Short-term option
        short_option = OptionPricing(self.S, self.K, 0.01, self.r, self.sigma, 'call')
        short_price, _, _, short_theta, _ = short_option.black_scholes()
        
        # Long-term option
        long_option = OptionPricing(self.S, self.K, 1.0, self.r, self.sigma, 'call')
        long_price, _, _, long_theta, _ = long_option.black_scholes()
        
        # Both thetas should be negative (time decay)
        self.assertLess(short_theta, 0)
        self.assertLess(long_theta, 0)
        
        # Long-term option should be more expensive
        self.assertGreater(long_price, short_price)
    
    def test_volatility_impact(self):
        """
        Test that option prices increase with volatility
        """
        # Low volatility option
        low_vol_option = OptionPricing(self.S, self.K, self.T, self.r, 0.1, 'call')
        low_vol_price, _, _, _, low_vol_vega = low_vol_option.black_scholes()
        
        # High volatility option
        high_vol_option = OptionPricing(self.S, self.K, self.T, self.r, 0.4, 'call')
        high_vol_price, _, _, _, high_vol_vega = high_vol_option.black_scholes()
        
        # High volatility option should be more expensive
        self.assertGreater(high_vol_price, low_vol_price)
        
        # Both should have positive vega
        self.assertGreater(low_vol_vega, 0)
        self.assertGreater(high_vol_vega, 0)
    
    def test_compare_models(self):
        """
        Test the compare_models method
        """
        results = self.call_option.compare_models()
        
        # Check that results DataFrame has correct columns
        expected_columns = ['Model', 'Option_Price', 'Standard_Error', 'Delta', 'Gamma', 'Theta', 'Vega']
        self.assertEqual(list(results.columns), expected_columns)
        
        # Check that all models are present
        models = results['Model'].tolist()
        self.assertIn('Black-Scholes', models)
        self.assertIn('Binomial', models)
        self.assertIn('Monte Carlo', models)
        
        # Check that all prices are positive
        self.assertTrue(np.all(results['Option_Price'] > 0))
        
        # Check that Monte Carlo has standard error
        mc_row = results[results['Model'] == 'Monte Carlo']
        self.assertGreater(mc_row['Standard_Error'].iloc[0], 0)


class TestEdgeCases(unittest.TestCase):
    """
    Test edge cases and boundary conditions
    """
    
    def test_zero_time_to_expiration(self):
        """
        Test options with zero time to expiration
        """
        option = OptionPricing(100, 105, 0, 0.05, 0.2, 'call')
        price, delta, gamma, theta, vega = option.black_scholes()
        
        # At expiration, call option should be max(S-K, 0)
        expected_price = max(100 - 105, 0)  # Should be 0 for OTM
        self.assertAlmostEqual(price, expected_price, places=10)
    
    def test_very_long_time_to_expiration(self):
        """
        Test options with very long time to expiration
        """
        option = OptionPricing(100, 105, 10, 0.05, 0.2, 'call')
        price, delta, gamma, theta, vega = option.black_scholes()
        
        # Long-term options should be expensive
        self.assertGreater(price, 40)  # Should be quite valuable
        
        # Delta should be close to 1 for long-term calls
        self.assertGreater(delta, 0.8)
    
    def test_zero_volatility(self):
        """
        Test options with zero volatility
        """
        option = OptionPricing(100, 105, 0.25, 0.05, 0, 'call')
        price, delta, gamma, theta, vega = option.black_scholes()
        
        # With zero volatility, option should behave like forward
        # For OTM call with zero volatility, price should be max(S*e^(rT) - K, 0)
        forward_price = max(100 * np.exp(0.05 * 0.25) - 105, 0)
        self.assertAlmostEqual(price, forward_price, places=10)
    
    def test_very_high_volatility(self):
        """
        Test options with very high volatility
        """
        option = OptionPricing(100, 105, 0.25, 0.05, 1.0, 'call')  # 100% volatility
        price, delta, gamma, theta, vega = option.black_scholes()
        
        # High volatility should make option expensive
        self.assertGreater(price, 15)
        
        # Vega should be very high
        self.assertGreater(vega, 10)


def run_tests():
    """
    Run all tests
    """
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestOptionPricing))
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestEdgeCases))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("Running Option Pricing Model Tests")
    print("=" * 40)
    
    success = run_tests()
    
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
