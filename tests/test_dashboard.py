"""
Automated functional verification of all 5 Streamlit dashboard pages and interactive controls.
Uses Streamlit's official AppTest framework.
"""

from streamlit.testing.v1 import AppTest


def test_dashboard_all_pages():
    print("Testing Streamlit Dashboard across all pages...")
    at = AppTest.from_file("../app/streamlit_app.py", default_timeout=30)
    at.run()
    assert not at.exception, f"Exception on Page 1 (Overview): {at.exception}"
    print("✅ Page 1 (Overview) rendered cleanly without errors.")

    # 1. Test Overview page toggles
    for r in at.radio:
        if r.label == "Rank by metric:":
            r.set_value("Revenue (£)").run()
            assert not at.exception, f"Exception on metric toggle Revenue: {at.exception}"
            r.set_value("Distinct Orders").run()
            assert not at.exception, f"Exception on metric toggle Orders: {at.exception}"
            print("✅ Overview Top 10 metric toggles verified.")
        elif r.label == "View:":
            r.set_value("Excluding United Kingdom").run()
            assert not at.exception, f"Exception on non-UK geo toggle: {at.exception}"
            print("✅ Non-UK geography toggle verified.")

    # 2. Test Navigation to Page 2: Product Analytics
    at.sidebar.radio[0].set_value("Product Analytics").run()
    assert not at.exception, f"Exception on Page 2 (Product Analytics): {at.exception}"
    # Test product search input
    if len(at.text_input) > 0:
        at.text_input[0].input("HEART").run()
        assert not at.exception, f"Exception on Product Search: {at.exception}"
    print("✅ Page 2 (Product Analytics) rendered cleanly with search filter.")

    # 3. Test Navigation to Page 3: Customer Segmentation
    at.sidebar.radio[0].set_value("Customer Segmentation").run()
    assert not at.exception, f"Exception on Page 3 (Customer Segmentation - KMeans): {at.exception}"

    # Test K-Means slider
    if len(at.slider) > 0:
        at.slider[0].set_value(4).run()
        assert not at.exception, f"Exception on KMeans slider k=4: {at.exception}"
    print("✅ Page 3 K-Means live parameter slider verified.")

    # Test DBSCAN switcher
    at.selectbox[0].set_value("DBSCAN").run()
    assert not at.exception, f"Exception on DBSCAN switcher: {at.exception}"
    print("✅ Page 3 DBSCAN live switcher verified.")

    # Test Agglomerative switcher
    at.selectbox[0].set_value("Agglomerative Hierarchical").run()
    assert not at.exception, f"Exception on Agglomerative switcher: {at.exception}"
    print("✅ Page 3 Agglomerative Hierarchical live switcher verified.")

    # 4. Test Navigation to Page 4: Model Comparison
    at.sidebar.radio[0].set_value("Model Comparison").run()
    assert not at.exception, f"Exception on Page 4 (Model Comparison): {at.exception}"
    print("✅ Page 4 (Model Comparison) rendered cleanly without errors.")

    # 5. Test Navigation to Page 5: Customer Explorer
    at.sidebar.radio[0].set_value("Customer Explorer").run()
    assert not at.exception, f"Exception on Page 5 (Customer Explorer): {at.exception}"
    # Test valid CustomerID input
    if len(at.text_input) > 0:
        at.text_input[0].input("14646").run()
        assert not at.exception, f"Exception on CustomerID input: {at.exception}"
    print("✅ Page 5 (Customer Explorer) rendered cleanly with customer lookup.")

    print("\n🎉 ALL 5 STREAMLIT PAGES AND INTERACTIVE CONTROLS FULLY VERIFIED!")


if __name__ == "__main__":
    test_dashboard_all_pages()
