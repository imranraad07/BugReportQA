import unittest
from preprocessing import clear_text


class MyTestCase(unittest.TestCase):
    def test_camel_case(self):
        text = "Some bug report having CamelCase mentions, in differentFormats ABCDtest things out. Method()!"
        text = clear_text(text).strip()
        self.assertEqual(
            "some bug report having camel case camelcase mentions in different formats differentformats abc dtest abcdtest things out method",
            text)


if __name__ == '__main__':
    unittest.main()
