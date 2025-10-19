def test_import_package():
    from amie.core import constants

    print(constants.DEFAULT_TRADE_NOTIONAL) # 1000.0


def test_nothing():
    assert True
    print("Smoke test passed.")

# if __name__ == "__main__":
#     test_import_package()
#     test_nothing()

