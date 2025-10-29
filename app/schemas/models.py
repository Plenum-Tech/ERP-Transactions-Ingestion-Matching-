# app/models.py
from sqlalchemy import (
    MetaData, Table, Column, String, Date, Time, Numeric, DateTime, text, UniqueConstraint, Index , ForeignKey
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
import uuid

# correct schema
metadata = MetaData(schema="erp_ai")

# helpers
def uuid_pk():
    return Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

def common_meta():
    return [
        Column("model_id", String, nullable=False),       # e.g., prebuilt-invoice, prebuilt-idDocument
        Column("source_filename", String),
        Column("source_url", String),
        Column("content_type", String),
        Column("created_at", DateTime(timezone=True), server_default=text("now()")),
    ]


# -----------------------------------------------------------
# BANK STATEMENT (from bank_statement.bank_stmt_to_dict: fields + accounts[])
# -----------------------------------------------------------
bank_statement_docs = Table(
    "bank_statement_docs",
    metadata,
    uuid_pk(),
    *common_meta(),
    # header fields
    Column("account_holder_name", String),
    Column("account_holder_address", String),
    Column("bank_name", String),
    Column("bank_address", String),
    Column("statement_start_date", Date),
    Column("statement_end_date", Date),
    # full shaped payloads
    Column("fields", JSONB, nullable=False, server_default=text("'{}'::jsonb")),
    Column("accounts", JSONB),  # list of accounts, each with balances and Transactions[]
)

# -----------------------------------------------------------
# RECEIPT (from receipt.receipt_to_dict: fields + items[])
# -----------------------------------------------------------
receipt_docs = Table(
    "receipt_docs",
    metadata,
    uuid_pk(),
    *common_meta(),
    # flattened header fields (align with _HEADER_FIELDS in receipt module)
    Column("merchant_name", String),
    Column("merchant_phone_number", String),
    Column("merchant_address", String),
    Column("merchant_address_recipient", String),
    Column("transaction_date", Date),
    Column("transaction_time", Time),
    Column("total", Numeric),
    Column("subtotal", Numeric),
    Column("total_tax", Numeric),
    Column("tip", Numeric),
    Column("currency", String),
    Column("payment_method", String),
    Column("card_number", String),
    Column("receipt_type", String),
    Column("customer_name", String),
    Column("order_number", String),
    Column("invoice_id", String),
    # full shaped payloads
    Column("fields", JSONB, nullable=False, server_default=text("'{}'::jsonb")),
    Column("items", JSONB),  # array of items: Description, Quantity, Price, TotalPrice, ProductCode, Date, Category
)


transactions_staging = Table(
    "transactions_staging",
    metadata,
    uuid_pk(),
    *common_meta(),  # keeps provenance cols like model_id, source_filename, etc. (as in your project)
    Column("txn_date", Date),
    Column("amount", Numeric(18, 2)),
    Column("currency", String),
    Column("merchant_raw", String),
    Column("merchant_norm", String(256)),
    Column("reference", String(128)),
    Column("source_file", String),               # optional: where we got it (URL/filename)
    Column("fingerprint", String(64), nullable=False),  # sha256 hex
    UniqueConstraint("fingerprint", name="uq_transactions_staging_fingerprint"),
    Index("ix_tx_stage_date", "txn_date"),
    Index("ix_tx_stage_merchant", "merchant_norm"),
    schema="erp_ai",
)

# -----------------------------------------------------------
# RECEIPTS STAGING (normalized header from receipts)
# -----------------------------------------------------------
receipts_staging = Table(
    "receipts_staging",
    metadata,
    uuid_pk(),
    *common_meta(),
    Column("receipt_date", Date),
    Column("total_amount", Numeric(18, 2)),
    Column("currency", String),
    Column("merchant_name", String),
    Column("merchant_norm", String(256)),
    Column("source_file", String),
    Column("fingerprint", String(64), nullable=False),  # sha256 hex
    UniqueConstraint("fingerprint", name="uq_receipts_staging_fingerprint"),
    Index("ix_rcpt_stage_date", "receipt_date"),
    Index("ix_rcpt_stage_merchant", "merchant_norm"),
    schema="erp_ai",
)


receipt_items_staging = Table(
    "receipt_items_staging",
    metadata,
    uuid_pk(),
    Column("receipt_id", UUID(as_uuid=True), ForeignKey("erp_ai.receipts_staging.id", ondelete="SET NULL"), nullable=True),
    Column("receipt_fingerprint", String(64), nullable=False),

    Column("line_no",  Numeric(10,0), nullable=False),
    Column("description", String),
    Column("quantity", Numeric(18,4)),
    Column("unit_price", Numeric(18,4)),
    Column("line_total", Numeric(18,4)),
    Column("product_code", String),
    Column("line_date", Date),
    Column("category", String),
    Column("currency", String(8)),
    Column("source_file", String),

    Column("item_fingerprint", String(64), nullable=False),
    Column("created_at", DateTime(timezone=True), server_default=text("now()")),

    UniqueConstraint("item_fingerprint", name="uq_receipt_items_item_fp"),
    Index("ix_rcpt_items_receipt_fp", "receipt_fingerprint"),
    Index("ix_rcpt_items_line_no", "line_no"),
    Index("ix_rcpt_items_product", "product_code"),
    schema="erp_ai",
)