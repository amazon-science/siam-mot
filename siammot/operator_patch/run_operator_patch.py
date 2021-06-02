# Patch the operators in maskrcnn_benchmark
import logging

logger = logging.getLogger(__name__)

try:
    import siammot.operator_patch.rpn_patch
except:
    logger.info("Error patching RPN operator")

try:
    import siammot.operator_patch.fpn_patch
except:
    logger.info("Error patching FPN operator")

logger.info("Operators from maskrcnn_benchmark are patched successfully!")

# Please don't patch your operators over here, because it can have unintended consequence
# unless you are sure about the consequences.
# Besides, do not change the patching order of the above operators, otherwise, the patching
# will fail even though it prints out the message that the the operators are patched successfully.
