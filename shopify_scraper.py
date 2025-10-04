import shopify

# Your Shopify credentials
SHOP_NAME = "your-shop-name"  # e.g., shopvasuki
API_KEY = "your-api-key"
ADMIN_API_ACCESS_TOKEN = "your-admin-api-access-token" # The one starting with shpat_

# Construct the shop URL
shop_url = f"https://{API_KEY}:{ADMIN_API_ACCESS_TOKEN}@{SHOP_NAME}.myshopify.com/admin"

# Activate the Shopify API session
shopify.ShopifyResource.set_site(shop_url)

# Fetch all products
products = shopify.Product.find()

# Print product data
for product in products:
    print(f"Title: {product.title}")
    print(f"Description: {product.body_html}")
    print(f"Price: {product.variants[0].price}")
    print("-" * 20)